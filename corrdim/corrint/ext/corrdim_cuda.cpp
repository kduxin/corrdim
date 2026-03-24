#include <torch/extension.h>

#include <vector>

torch::Tensor correlation_counts_self_cuda(torch::Tensor vecs, torch::Tensor log_eps);
torch::Tensor correlation_counts_cross_cuda(torch::Tensor vecs1, torch::Tensor vecs2, torch::Tensor log_eps);
torch::Tensor progressive_counts_self_cuda(torch::Tensor vecs, torch::Tensor log_eps);

std::vector<int64_t> shape2(const torch::Tensor& x) {
  auto s = x.sizes();
  return {static_cast<int64_t>(s[0]), static_cast<int64_t>(s[1])};
}

torch::Tensor correlation_counts_self(torch::Tensor vecs, torch::Tensor log_eps) {
  TORCH_CHECK(vecs.is_cuda(), "vecs must be CUDA tensor");
  TORCH_CHECK(log_eps.is_cuda(), "log_eps must be CUDA tensor");
  TORCH_CHECK(vecs.dim() == 3, "vecs must be (B, M, K)");
  TORCH_CHECK(log_eps.dim() == 1, "log_eps must be (T,)");
  return correlation_counts_self_cuda(vecs, log_eps);
}

torch::Tensor correlation_counts_cross(torch::Tensor vecs1, torch::Tensor vecs2, torch::Tensor log_eps) {
  TORCH_CHECK(vecs1.is_cuda() && vecs2.is_cuda(), "vecs1/vecs2 must be CUDA tensors");
  TORCH_CHECK(log_eps.is_cuda(), "log_eps must be CUDA tensor");
  TORCH_CHECK(vecs1.dim() == 3 && vecs2.dim() == 3, "vecs1/vecs2 must be (B, M, K)/(B, N, K)");
  TORCH_CHECK(log_eps.dim() == 1, "log_eps must be (T,)");
  TORCH_CHECK(vecs1.size(0) == vecs2.size(0), "batch size mismatch");
  TORCH_CHECK(vecs1.size(2) == vecs2.size(2), "feature dim mismatch");
  return correlation_counts_cross_cuda(vecs1, vecs2, log_eps);
}

torch::Tensor progressive_counts_self(torch::Tensor vecs, torch::Tensor log_eps) {
  TORCH_CHECK(vecs.is_cuda(), "vecs must be CUDA tensor");
  TORCH_CHECK(log_eps.is_cuda(), "log_eps must be CUDA tensor");
  TORCH_CHECK(vecs.dim() == 3, "vecs must be (B, M, K)");
  TORCH_CHECK(log_eps.dim() == 1, "log_eps must be (T,)");
  return progressive_counts_self_cuda(vecs, log_eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("correlation_counts_self", &correlation_counts_self, "Self correlation counts (CUDA)");
  m.def("correlation_counts_cross", &correlation_counts_cross, "Cross correlation counts (CUDA)");
  m.def("progressive_counts_self", &progressive_counts_self, "Progressive self counts (CUDA)");
}

