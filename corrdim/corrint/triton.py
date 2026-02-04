import torch
import numpy as np
import triton
import triton.language as tl

__all__ = [
    'correlation_counts',
    'correlation_integral',
    'progressive_correlation_counts',
    'progressive_correlation_integral',
]

_RESOURCE_EXHAUSTED_MARKERS = (
    "out of resources",
    "too many resources requested",
    "shared memory",
    "local memory",
    "invalid configuration argument",
)


def _is_resource_exhausted_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _RESOURCE_EXHAUSTED_MARKERS)


def _select_block_sizes(device: torch.device, *, m: int, n: int | None, k: int):
    props = torch.cuda.get_device_properties(device)
    shared_mem = getattr(props, "shared_memory_per_block", 0) or 0

    block_mn = 64
    block_k = 32
    group_m = 8
    if shared_mem and shared_mem < 64 * 1024:
        block_mn = 32
        block_k = 16
        group_m = 4

    def _pow2_cap(value: int, cap: int) -> int:
        return min(cap, 2 ** int(np.ceil(np.log2(max(value, 1)))))

    block_m = _pow2_cap(m, block_mn)
    block_n = _pow2_cap(n, block_mn) if n is not None else None
    block_k = _pow2_cap(k, block_k)

    return block_m, block_n, block_k, group_m


def _candidate_block_sizes(device: torch.device, *, m: int, n: int | None, k: int):
    block_m, block_n, block_k, group_m = _select_block_sizes(device, m=m, n=n, k=k)

    def _pow2_floor(value: int) -> int:
        return 2 ** int(np.floor(np.log2(max(value, 1))))

    def _downscale(value: int | None, factor: int, min_value: int) -> int | None:
        if value is None:
            return None
        return max(min_value, _pow2_floor(value // factor))

    candidates = [(block_m, block_n, block_k, group_m)]
    for factor, min_mn, min_k, min_group in ((2, 16, 16, 4), (4, 16, 8, 2)):
        scaled_m = _downscale(block_m, factor, min_mn)
        scaled_n = _downscale(block_n, factor, min_mn) if block_n is not None else None
        scaled_k = _downscale(block_k, factor, min_k)
        scaled_group = max(min_group, group_m // factor)
        entry = (scaled_m, scaled_n, scaled_k, scaled_group)
        if entry not in candidates:
            candidates.append(entry)

    return candidates

@triton.jit
def batched_correlation_counts_cross_kernel(
    vecs1_ptr, vecs2_ptr, logthreshs_ptr, counts_ptr,
    B, M, N, K, T: tl.constexpr,
    stride_b1, stride_m1, stride_k1,
    stride_b2, stride_n2, stride_k2,
    stride_t,
    stride_cb, stride_ct,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    same_vecs: tl.constexpr,
):
    """Batched version of correlation integral counting.
    Each program handles a (BLOCK_SIZE_M x BLOCK_SIZE_N) block within one batch index.
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    total_pid = B * num_pid_in_batch
    if pid >= total_pid:
        return

    pid_b = pid // num_pid_in_batch
    pid_local = pid - pid_b * num_pid_in_batch

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_local // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_local % num_pid_in_group) % group_size_m)
    pid_n = (pid_local % num_pid_in_group) // group_size_m

    if same_vecs and (pid_m > pid_n):
        return

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_m1 > 0)
    tl.assume(stride_n2 > 0)
    tl.assume(stride_k1 > 0)
    tl.assume(stride_k2 > 0)
    tl.assume(stride_t > 0)
    tl.assume(stride_ct > 0)

    if same_vecs:
        tl.assume(BLOCK_SIZE_M == BLOCK_SIZE_N)
        tl.assume(M == N)

    base_a = vecs1_ptr + pid_b * stride_b1
    base_b = vecs2_ptr + pid_b * stride_b2
    base_c = counts_ptr + pid_b * stride_cb

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = base_a + (offs_am[:, None] * stride_m1 + offs_k[None, :] * stride_k1)
    b_ptrs = base_b + (offs_bn[:, None] * stride_n2 + offs_k[None, :] * stride_k2)
    offs_am_mask = offs_am < M
    offs_bn_mask = offs_bn < N

    distsq = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offs < K

        a = tl.load(a_ptrs, mask=offs_am_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=offs_bn_mask[:, None] & k_mask[None, :], other=0.0)
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        a2 = tl.sum(a * a, axis=1)
        b2 = tl.sum(b * b, axis=1)
        distsq_delta = tl.dot(-2 * a, b.T, acc=a2[:, None] + b2[None, :], input_precision='ieee')
        distsq += distsq_delta

        a_ptrs += BLOCK_SIZE_K * stride_k1
        b_ptrs += BLOCK_SIZE_K * stride_k2

    mask = offs_am_mask[:, None] & offs_bn_mask[None, :]
    if same_vecs and pid_m == pid_n:
        triu_mask = tl.arange(0, BLOCK_SIZE_M)[:, None] < tl.arange(0, BLOCK_SIZE_M)[None, :]
        mask = mask & triu_mask
    distsq = tl.where(mask, distsq, torch.inf)
    logdist = tl.where(distsq > 0, 0.5 * tl.log(distsq), -torch.inf)

    for i in range(0, T):
        logthresh = tl.load(logthreshs_ptr + i * stride_t)
        count = tl.sum((logdist <= logthresh).to(tl.int64))
        if same_vecs:
            count *= 2
        tl.atomic_add(base_c + i * stride_ct, count.to(tl.int64))


@triton.jit
def batched_progressive_counts_kernel(
    vecs_ptr,
    logthreshs_ptr,
    inc_ptr,
    B,
    M,
    K,
    T: tl.constexpr,
    stride_b,
    stride_m,
    stride_k,
    stride_t,
    stride_ib,
    stride_im,
    stride_it,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Compute per-position incremental counts for progressive correlation counts.

    For each column index j, this kernel accumulates the number of i < j such that
    dist(vecs[i], vecs[j]) <= epsilon, for every epsilon in logthreshs_ptr.

    The output `inc` has shape (B, M, T), where inc[b, j, t] stores the incremental
    count for position j and threshold t (unordered pairs, i < j).
    The caller can then compute:
        counts = 2 * cumsum(inc, dim=1)
    to match the ordered-pair semantics used by `correlation_counts(..., vecs_other=None)`.
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = num_pid_m
    num_pid_in_batch = num_pid_m * num_pid_n
    total_pid = B * num_pid_in_batch
    if pid >= total_pid:
        return

    pid_b = pid // num_pid_in_batch
    pid_local = pid - pid_b * num_pid_in_batch

    # Grouped ordering to improve L2 reuse (same scheme as correlation_counts kernel).
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_local // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_local % num_pid_in_group) % group_size_m)
    pid_n = (pid_local % num_pid_in_group) // group_size_m

    # Only compute upper triangle blocks (including diagonal).
    if pid_m > pid_n:
        return

    base_v = vecs_ptr + pid_b * stride_b
    base_i = inc_ptr + pid_b * stride_ib

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = base_v + (offs_am[:, None] * stride_m + offs_k[None, :] * stride_k)
    b_ptrs = base_v + (offs_bn[:, None] * stride_m + offs_k[None, :] * stride_k)

    offs_am_mask = offs_am < M
    offs_bn_mask = offs_bn < M

    distsq = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offs < K

        a = tl.load(a_ptrs, mask=offs_am_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=offs_bn_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        a2 = tl.sum(a * a, axis=1)
        b2 = tl.sum(b * b, axis=1)
        distsq += tl.dot(-2 * a, b.T, acc=a2[:, None] + b2[None, :], input_precision="ieee")

        a_ptrs += BLOCK_SIZE_K * stride_k
        b_ptrs += BLOCK_SIZE_K * stride_k

    mask = offs_am_mask[:, None] & offs_bn_mask[None, :]
    if pid_m == pid_n:
        # Strictly upper triangle only (i < j) within a diagonal block.
        triu_mask = tl.arange(0, BLOCK_SIZE_M)[:, None] < tl.arange(0, BLOCK_SIZE_M)[None, :]
        mask = mask & triu_mask
    distsq = tl.where(mask, distsq, torch.inf)
    logdist = tl.where(distsq > 0, 0.5 * tl.log(distsq), -torch.inf)

    # Atomically accumulate per-column counts into inc[b, j, t].
    for t in range(0, T):
        logthresh = tl.load(logthreshs_ptr + t * stride_t)
        active = (logdist <= logthresh).to(tl.int64)
        col_counts = tl.sum(active, axis=0).to(tl.int64)  # (BLOCK_SIZE_M,)
        tl.atomic_add(
            base_i + offs_bn * stride_im + t * stride_it,
            col_counts,
            mask=offs_bn_mask,
        )


def _batched_progressive_correlation_counts(vecs: torch.FloatTensor, epsilons: torch.FloatTensor) -> torch.LongTensor:
    assert vecs.dim() == 3, "Input must be (B, M, K)"
    B, M, K = vecs.shape
    T = epsilons.shape[0]
    log_epsilons = epsilons.log()

    # inc[b, j, t] = number of i < j with dist(i, j) <= eps[t]
    inc = torch.zeros((B, M, T), device=vecs.device, dtype=torch.int64)

    grid = lambda META: (B * triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(M, META["BLOCK_SIZE_M"]),)
    torch.cuda.set_device(vecs.device)
    for block_m, _block_n, block_k, group_m in _candidate_block_sizes(vecs.device, m=M, n=None, k=K):
        try:
            inc.zero_()
            batched_progressive_counts_kernel[grid](
                vecs,
                log_epsilons,
                inc,
                B,
                M,
                K,
                T,
                vecs.stride(0),
                vecs.stride(1),
                vecs.stride(2),
                log_epsilons.stride(0),
                inc.stride(0),
                inc.stride(1),
                inc.stride(2),
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_K=block_k,
                GROUP_SIZE_M=group_m,
            )
            torch.cuda.synchronize()
            break
        except RuntimeError as exc:
            if not _is_resource_exhausted_error(exc):
                raise
    else:
        from . import pytorch as torch_impl

        return torch_impl.progressive_correlation_counts(vecs, epsilons)

    # Convert incremental unordered counts into prefix ordered-pair counts.
    return inc.cumsum(dim=1) * 2



def _batched_correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
) -> torch.LongTensor:
    """Counts, per batch, the number of pairwise distances <= epsilon for a set of vectors."""
    assert vecs.dim() == 3, "Input must be (B, M, K)"

    B, M, K = vecs.shape
    T = epsilons.shape[0]
    log_epsilons = epsilons.log()
    counts = torch.zeros((B, T), device=vecs.device, dtype=torch.int64)

    grid = lambda META: (
        B * triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(M, META['BLOCK_SIZE_N']),
    )
    torch.cuda.set_device(vecs.device)
    for block_m, block_n, block_k, group_m in _candidate_block_sizes(vecs.device, m=M, n=M, k=K):
        try:
            counts.zero_()
            batched_correlation_counts_cross_kernel[grid](
                vecs, vecs, log_epsilons, counts,
                B, M, M, K, T,
                vecs.stride(0), vecs.stride(1), vecs.stride(2),
                vecs.stride(0), vecs.stride(1), vecs.stride(2),
                log_epsilons.stride(0),
                counts.stride(0), counts.stride(1),
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                GROUP_SIZE_M=group_m,
                same_vecs=True,
            )
            torch.cuda.synchronize()
            break
        except RuntimeError as exc:
            if not _is_resource_exhausted_error(exc):
                raise
    else:
        from . import pytorch as torch_impl

        return torch_impl.correlation_counts(vecs, epsilons)
    return counts

def _batched_correlation_counts_cross(
    vecs1: torch.FloatTensor,
    vecs2: torch.FloatTensor,
    epsilons: torch.FloatTensor,
) -> torch.LongTensor:
    """Counts, per batch, the number of cross-distances <= epsilon between vecs1 and vecs2."""
    if vecs1 is vecs2:
        return _batched_correlation_counts(vecs1, epsilons)

    assert vecs1.dim() == 3 and vecs2.dim() == 3, "Inputs must be (B, M, K) and (B, N, K)"

    B1, M, K1 = vecs1.shape
    B2, N, K2 = vecs2.shape
    assert B1 == B2, "Batch sizes must match"
    assert K1 == K2, "Vector dimensions must match"

    T = epsilons.shape[0]
    log_epsilons = epsilons.log()
    counts = torch.zeros((B1, T), device=vecs1.device, dtype=torch.int64)

    grid = lambda META: (
        B1 * triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    torch.cuda.set_device(vecs1.device)
    for block_m, block_n, block_k, group_m in _candidate_block_sizes(vecs1.device, m=M, n=N, k=K1):
        batched_block_m = min(block_m, 2 ** int(np.ceil(np.log2(M))))
        batched_block_n = min(block_n, 2 ** int(np.ceil(np.log2(N))))
        try:
            counts.zero_()
            batched_correlation_counts_cross_kernel[grid](
                vecs1, vecs2, log_epsilons, counts,
                B1, M, N, K1, T,
                vecs1.stride(0), vecs1.stride(1), vecs1.stride(2),
                vecs2.stride(0), vecs2.stride(1), vecs2.stride(2),
                log_epsilons.stride(0),
                counts.stride(0), counts.stride(1),
                BLOCK_SIZE_M=batched_block_m,
                BLOCK_SIZE_N=batched_block_n,
                BLOCK_SIZE_K=block_k,
                GROUP_SIZE_M=group_m,
                same_vecs=False,
            )
            torch.cuda.synchronize()
            break
        except RuntimeError as exc:
            if not _is_resource_exhausted_error(exc):
                raise
    else:
        from . import pytorch as torch_impl

        return torch_impl.correlation_counts(vecs1, epsilons, vecs_other=vecs2)
    return counts


def correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    vecs_other: torch.FloatTensor = None,
) -> torch.LongTensor:
    """
    Compute the correlation integral counts (batched) between two sequences of vectors. For each epsilon in epsilons, counts the number of vector pairs (across the two inputs) whose distance is less than or equal to epsilon.
    - (i, j) and (j, i) are counted twice. 
    - If vecs_other is not provided, computes all pairs (i, j) within vecs, excluding the diagonal elements (i, i).
    - For epsilon=infty, counts would be M * (M - 1) if vecs_other is not provided, and M * N if vecs_other is provided.
    
    Args:
        vecs: Input tensor of shape (M, K) or (B, M, K).
        epsilons: 1D tensor of length T.
        vecs_other: Optional tensor of shape (N, K) or (B, N, K). Defaults to vecs if not provided.

    Returns:
        torch.LongTensor: Correlation counts for each batch (if batched) and each epsilon value.
    """
    ndim = vecs.dim()
    assert ndim in [2, 3], "Input must be (M, K) or (B, M, K)"
    if vecs_other is None:
        if ndim == 2:
            vecs = vecs.unsqueeze(0)
        counts = _batched_correlation_counts(vecs, epsilons)
        if ndim == 2:
            counts = counts.squeeze(0)
        return counts

    assert vecs.dim() == vecs_other.dim()
    if ndim == 2:
        vecs = vecs.unsqueeze(0)
        vecs_other = vecs_other.unsqueeze(0)
    counts = _batched_correlation_counts_cross(vecs, vecs_other, epsilons)
    if ndim == 2:
        counts = counts.squeeze(0)

    return counts


def correlation_integral(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    vecs_other: torch.FloatTensor = None,
) -> torch.FloatTensor:
    ndim = vecs.dim()
    assert ndim in [2, 3], "Input must be (M, K) or (B, M, K)"
    if vecs_other is None:
        counts = correlation_counts(vecs, epsilons)
        M = vecs.shape[-2]
        return counts.to(torch.float32) / (M * (M - 1))
    else:
        counts = correlation_counts(vecs, epsilons, vecs_other)
        M, N = vecs.shape[-2], vecs_other.shape[-2]
        return counts.to(torch.float32) / (M * N)


def progressive_correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
) -> torch.LongTensor:
    """Compute the counts of V[:t] at every step t of the vector sequence V.
    In implementation, the counts is updated at every step t by adding
    an additional count of V[t] with every vector in V[:t].
    This function supports batched computation.

    vecs: (M, K) or (B, M, K)
    epsilons: (T,)
    """
    ndim = vecs.dim()
    if ndim == 2:
        vecs = vecs.unsqueeze(0)
    counts = _batched_progressive_correlation_counts(vecs, epsilons)
    if ndim == 2:
        counts = counts.squeeze(0)
    return counts

def progressive_correlation_integral(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
) -> torch.FloatTensor:
    """Compute the progressive correlation integral of V[:t] at every step t of the vector sequence V.
    In implementation, the integral is updated at every step t by adding
    an additional integral of V[t] with every vector in V[:t].
    This function supports batched computation.

    vecs: (M, K) or (B, M, K)
    epsilons: (T,)
    """
    counts = progressive_correlation_counts(vecs, epsilons)
    # (B, M, T)

    M = vecs.shape[-2]
    pairs = (
        torch.arange(0, M, device=vecs.device, dtype=torch.float32) 
        * torch.arange(1, M+1, device=vecs.device, dtype=torch.float32)
    )
    pairs[0] = 1e-6
    return counts.to(torch.float32) / pairs.unsqueeze(-1)
