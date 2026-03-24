#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int TILE = 16;
constexpr int BK = 16;

__device__ __forceinline__ void atomic_add_i64(int64_t* addr, int64_t v) {
  atomicAdd(reinterpret_cast<unsigned long long*>(addr), static_cast<unsigned long long>(v));
}

__device__ __forceinline__ int64_t lower_bound_log_eps(
    const float* __restrict__ log_eps,
    int64_t T,
    float logdist) {
  int64_t lo = 0;
  int64_t hi = T;
  while (lo < hi) {
    int64_t mid = lo + ((hi - lo) >> 1);
    if (log_eps[mid] < logdist) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename scalar_t>
__global__ void sqnorms_kernel(
    const scalar_t* __restrict__ vecs,
    float* __restrict__ norms,
    int64_t B,
    int64_t M,
    int64_t K) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = B * M;
  if (idx >= total) return;

  int64_t b = idx / M;
  int64_t i = idx - b * M;
  const scalar_t* vi = vecs + (b * M + i) * K;

  float acc = 0.0f;
  #pragma unroll 4
  for (int64_t k = 0; k < K; ++k) {
    float v = static_cast<float>(vi[k]);
    acc += v * v;  // IEEE fp32 accumulation
  }
  norms[idx] = acc;
}

template <typename scalar_t>
__global__ void cross_counts_tiled_kernel(
    const scalar_t* __restrict__ vecs1,
    const scalar_t* __restrict__ vecs2,
    const float* __restrict__ norms1,
    const float* __restrict__ norms2,
    const float* __restrict__ log_eps,
    int64_t* __restrict__ counts_diff,
    int64_t B,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t T) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * TILE;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * TILE;
  const int64_t i = row0 + ty;
  const int64_t j = col0 + tx;

  __shared__ float As[TILE][BK];
  __shared__ float Bs[TILE][BK];

  float dot = 0.0f;
  for (int64_t k0 = 0; k0 < K; k0 += BK) {
    const int tid = ty * TILE + tx;
    const int nthreads = TILE * TILE;

    for (int idx = tid; idx < TILE * BK; idx += nthreads) {
      const int r = idx / BK;
      const int c = idx - r * BK;
      const int64_t gi = row0 + r;
      const int64_t gk = k0 + c;
      As[r][c] = (gi < M && gk < K) ? static_cast<float>(vecs1[(b * M + gi) * K + gk]) : 0.0f;
    }
    for (int idx = tid; idx < TILE * BK; idx += nthreads) {
      const int r = idx / BK;
      const int c = idx - r * BK;
      const int64_t gj = col0 + r;
      const int64_t gk = k0 + c;
      Bs[r][c] = (gj < N && gk < K) ? static_cast<float>(vecs2[(b * N + gj) * K + gk]) : 0.0f;
    }
    __syncthreads();

    if (i < M && j < N) {
      #pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        dot += As[ty][kk] * Bs[tx][kk];  // IEEE fp32 dot accumulation
      }
    }
    __syncthreads();
  }

  if (i < M && j < N) {
    float d2 = norms1[b * M + i] + norms2[b * N + j] - 2.0f * dot;
    float logdist = (d2 > 0.0f) ? 0.5f * logf(d2) : -INFINITY;
    int64_t idx = lower_bound_log_eps(log_eps, T, logdist);
    if (idx < T) {
      int64_t* out = counts_diff + b * (T + 1);
      atomic_add_i64(out + idx, 1LL);
      atomic_add_i64(out + T, -1LL);
    }
  }
}

template <typename scalar_t>
__global__ void self_counts_tiled_kernel(
    const scalar_t* __restrict__ vecs,
    const float* __restrict__ norms,
    const float* __restrict__ log_eps,
    int64_t* __restrict__ counts_diff,
    int64_t B,
    int64_t M,
    int64_t K,
    int64_t T) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t row_blk = blockIdx.y;
  const int64_t col_blk = blockIdx.x;
  if (row_blk > col_blk) return;

  const int64_t row0 = row_blk * TILE;
  const int64_t col0 = col_blk * TILE;
  const int64_t i = row0 + ty;
  const int64_t j = col0 + tx;
  if (i >= M || j >= M || i >= j) return;

  __shared__ float As[TILE][BK];
  __shared__ float Bs[TILE][BK];

  float dot = 0.0f;
  for (int64_t k0 = 0; k0 < K; k0 += BK) {
    const int tid = ty * TILE + tx;
    const int nthreads = TILE * TILE;

    for (int idx = tid; idx < TILE * BK; idx += nthreads) {
      const int r = idx / BK;
      const int c = idx - r * BK;
      const int64_t gi = row0 + r;
      const int64_t gk = k0 + c;
      As[r][c] = (gi < M && gk < K) ? static_cast<float>(vecs[(b * M + gi) * K + gk]) : 0.0f;
    }
    for (int idx = tid; idx < TILE * BK; idx += nthreads) {
      const int r = idx / BK;
      const int c = idx - r * BK;
      const int64_t gj = col0 + r;
      const int64_t gk = k0 + c;
      Bs[r][c] = (gj < M && gk < K) ? static_cast<float>(vecs[(b * M + gj) * K + gk]) : 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      dot += As[ty][kk] * Bs[tx][kk];
    }
    __syncthreads();
  }

  float d2 = norms[b * M + i] + norms[b * M + j] - 2.0f * dot;
  float logdist = (d2 > 0.0f) ? 0.5f * logf(d2) : -INFINITY;
  int64_t idx = lower_bound_log_eps(log_eps, T, logdist);
  if (idx < T) {
    int64_t* out = counts_diff + b * (T + 1);
    atomic_add_i64(out + idx, 2LL);
    atomic_add_i64(out + T, -2LL);
  }
}

template <typename scalar_t>
__global__ void progressive_counts_tiled_kernel(
    const scalar_t* __restrict__ vecs,
    const float* __restrict__ norms,
    const float* __restrict__ log_eps,
    int64_t* __restrict__ inc_diff,
    int64_t B,
    int64_t M,
    int64_t K,
    int64_t T) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t row_blk = blockIdx.y;
  const int64_t col_blk = blockIdx.x;
  if (row_blk > col_blk) return;

  const int64_t row0 = row_blk * TILE;
  const int64_t col0 = col_blk * TILE;
  const int64_t i = row0 + ty;
  const int64_t j = col0 + tx;
  if (i >= M || j >= M || i >= j) return;

  __shared__ float As[TILE][BK];
  __shared__ float Bs[TILE][BK];

  float dot = 0.0f;
  for (int64_t k0 = 0; k0 < K; k0 += BK) {
    const int tid = ty * TILE + tx;
    const int nthreads = TILE * TILE;

    for (int idx = tid; idx < TILE * BK; idx += nthreads) {
      const int r = idx / BK;
      const int c = idx - r * BK;
      const int64_t gi = row0 + r;
      const int64_t gk = k0 + c;
      As[r][c] = (gi < M && gk < K) ? static_cast<float>(vecs[(b * M + gi) * K + gk]) : 0.0f;
    }
    for (int idx = tid; idx < TILE * BK; idx += nthreads) {
      const int r = idx / BK;
      const int c = idx - r * BK;
      const int64_t gj = col0 + r;
      const int64_t gk = k0 + c;
      Bs[r][c] = (gj < M && gk < K) ? static_cast<float>(vecs[(b * M + gj) * K + gk]) : 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      dot += As[ty][kk] * Bs[tx][kk];
    }
    __syncthreads();
  }

  float d2 = norms[b * M + i] + norms[b * M + j] - 2.0f * dot;
  float logdist = (d2 > 0.0f) ? 0.5f * logf(d2) : -INFINITY;
  int64_t idx = lower_bound_log_eps(log_eps, T, logdist);
  if (idx < T) {
    int64_t* out_j = inc_diff + (b * M + j) * (T + 1);
    atomic_add_i64(out_j + idx, 1LL);
    atomic_add_i64(out_j + T, -1LL);
  }
}

}  // namespace

torch::Tensor correlation_counts_self_cuda(torch::Tensor vecs, torch::Tensor log_eps) {
  const auto B = vecs.size(0);
  const auto M = vecs.size(1);
  const auto K = vecs.size(2);
  const auto T = log_eps.size(0);

  auto vecs_c = vecs.contiguous();
  auto eps_c = log_eps.contiguous().to(torch::kFloat32);
  auto norms = torch::zeros({B, M}, vecs.options().dtype(torch::kFloat32));
  auto counts_diff = torch::zeros({B, T + 1}, vecs.options().dtype(torch::kInt64));

  c10::cuda::CUDAGuard device_guard(vecs.device());
  auto stream = at::cuda::getDefaultCUDAStream();

  constexpr int norm_threads = 256;
  dim3 norm_grid((B * M + norm_threads - 1) / norm_threads);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(vecs.scalar_type(), "sqnorms_self_kernel", [&] {
    sqnorms_kernel<scalar_t><<<norm_grid, norm_threads, 0, stream>>>(
        vecs_c.data_ptr<scalar_t>(), norms.data_ptr<float>(), B, M, K);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 block(TILE, TILE);
  dim3 grid((M + TILE - 1) / TILE, (M + TILE - 1) / TILE, B);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(vecs.scalar_type(), "self_counts_tiled_kernel", [&] {
    self_counts_tiled_kernel<scalar_t><<<grid, block, 0, stream>>>(
        vecs_c.data_ptr<scalar_t>(),
        norms.data_ptr<float>(),
        eps_c.data_ptr<float>(),
        counts_diff.data_ptr<int64_t>(),
        B, M, K, T);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto counts_prefix = torch::cumsum(counts_diff, -1);
  return counts_prefix.narrow(-1, 0, T);
}

torch::Tensor correlation_counts_cross_cuda(torch::Tensor vecs1, torch::Tensor vecs2, torch::Tensor log_eps) {
  const auto B = vecs1.size(0);
  const auto M = vecs1.size(1);
  const auto N = vecs2.size(1);
  const auto K = vecs1.size(2);
  const auto T = log_eps.size(0);

  auto v1 = vecs1.contiguous();
  auto v2 = vecs2.contiguous();
  auto eps_c = log_eps.contiguous().to(torch::kFloat32);
  auto norms1 = torch::zeros({B, M}, vecs1.options().dtype(torch::kFloat32));
  auto norms2 = torch::zeros({B, N}, vecs2.options().dtype(torch::kFloat32));
  auto counts_diff = torch::zeros({B, T + 1}, vecs1.options().dtype(torch::kInt64));

  c10::cuda::CUDAGuard device_guard(vecs1.device());
  auto stream = at::cuda::getDefaultCUDAStream();

  constexpr int norm_threads = 256;
  dim3 norm1_grid((B * M + norm_threads - 1) / norm_threads);
  dim3 norm2_grid((B * N + norm_threads - 1) / norm_threads);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(vecs1.scalar_type(), "sqnorms_cross_kernel", [&] {
    sqnorms_kernel<scalar_t><<<norm1_grid, norm_threads, 0, stream>>>(
        v1.data_ptr<scalar_t>(), norms1.data_ptr<float>(), B, M, K);
    sqnorms_kernel<scalar_t><<<norm2_grid, norm_threads, 0, stream>>>(
        v2.data_ptr<scalar_t>(), norms2.data_ptr<float>(), B, N, K);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, B);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(vecs1.scalar_type(), "cross_counts_tiled_kernel", [&] {
    cross_counts_tiled_kernel<scalar_t><<<grid, block, 0, stream>>>(
        v1.data_ptr<scalar_t>(),
        v2.data_ptr<scalar_t>(),
        norms1.data_ptr<float>(),
        norms2.data_ptr<float>(),
        eps_c.data_ptr<float>(),
        counts_diff.data_ptr<int64_t>(),
        B, M, N, K, T);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto counts_prefix = torch::cumsum(counts_diff, -1);
  return counts_prefix.narrow(-1, 0, T);
}

torch::Tensor progressive_counts_self_cuda(torch::Tensor vecs, torch::Tensor log_eps) {
  const auto B = vecs.size(0);
  const auto M = vecs.size(1);
  const auto K = vecs.size(2);
  const auto T = log_eps.size(0);

  auto vecs_c = vecs.contiguous();
  auto eps_c = log_eps.contiguous().to(torch::kFloat32);
  auto norms = torch::zeros({B, M}, vecs.options().dtype(torch::kFloat32));
  auto inc_diff = torch::zeros({B, M, T + 1}, vecs.options().dtype(torch::kInt64));

  c10::cuda::CUDAGuard device_guard(vecs.device());
  auto stream = at::cuda::getDefaultCUDAStream();

  constexpr int norm_threads = 256;
  dim3 norm_grid((B * M + norm_threads - 1) / norm_threads);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(vecs.scalar_type(), "sqnorms_progressive_kernel", [&] {
    sqnorms_kernel<scalar_t><<<norm_grid, norm_threads, 0, stream>>>(
        vecs_c.data_ptr<scalar_t>(), norms.data_ptr<float>(), B, M, K);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 block(TILE, TILE);
  dim3 grid((M + TILE - 1) / TILE, (M + TILE - 1) / TILE, B);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(vecs.scalar_type(), "progressive_counts_tiled_kernel", [&] {
    progressive_counts_tiled_kernel<scalar_t><<<grid, block, 0, stream>>>(
        vecs_c.data_ptr<scalar_t>(),
        norms.data_ptr<float>(),
        eps_c.data_ptr<float>(),
        inc_diff.data_ptr<int64_t>(),
        B, M, K, T);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto inc_prefix = torch::cumsum(inc_diff, -1);
  return inc_prefix.narrow(-1, 0, T);
}

