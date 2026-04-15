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


def _normalize_seq_lens(
    seq_lens: torch.Tensor | None,
    *,
    batch_size: int,
    max_len: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Return validated per-batch sequence lengths on `device` as int32."""
    if seq_lens is None:
        return torch.full((batch_size,), int(max_len), device=device, dtype=torch.int32)
    if not isinstance(seq_lens, torch.Tensor):
        seq_lens = torch.as_tensor(seq_lens)
    if seq_lens.dim() != 1 or seq_lens.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape (B,), got {tuple(seq_lens.shape)}.")
    out = seq_lens.to(device=device, dtype=torch.int32)
    if torch.any(out < 0):
        raise ValueError(f"{name} must be non-negative.")
    if torch.any(out > max_len):
        raise ValueError(f"{name} must be <= max sequence length {max_len}.")
    return out.contiguous()

@triton.jit
def batched_correlation_counts_cross_kernel(
    vecs1_ptr, vecs2_ptr, seq_lens1_ptr, seq_lens2_ptr, logthreshs_ptr, counts_ptr,
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
    m_b = tl.load(seq_lens1_ptr + pid_b).to(tl.int32)
    n_b = tl.load(seq_lens2_ptr + pid_b).to(tl.int32)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = base_a + (offs_am[:, None] * stride_m1 + offs_k[None, :] * stride_k1)
    b_ptrs = base_b + (offs_bn[:, None] * stride_n2 + offs_k[None, :] * stride_k2)
    offs_am_mask = offs_am < m_b
    offs_bn_mask = offs_bn < n_b

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
    seq_lens_ptr,
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
    m_b = tl.load(seq_lens_ptr + pid_b).to(tl.int32)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = base_v + (offs_am[:, None] * stride_m + offs_k[None, :] * stride_k)
    b_ptrs = base_v + (offs_bn[:, None] * stride_m + offs_k[None, :] * stride_k)

    offs_am_mask = offs_am < m_b
    offs_bn_mask = offs_bn < m_b

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


def _batched_progressive_correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    *,
    seq_lens: torch.Tensor | None = None,
) -> torch.LongTensor:
    assert vecs.dim() == 3, "Input must be (B, M, K)"
    B, M, K = vecs.shape
    T = epsilons.shape[0]
    log_epsilons = epsilons.log()
    seq_lens_i32 = _normalize_seq_lens(seq_lens, batch_size=B, max_len=M, device=vecs.device, name="seq_lens")

    # inc[b, j, t] = number of i < j with dist(i, j) <= eps[t]
    inc = torch.zeros((B, M, T), device=vecs.device, dtype=torch.int64)

    grid = lambda META: (B * triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(M, META["BLOCK_SIZE_M"]),)
    torch.cuda.set_device(vecs.device)
    for block_m, _block_n, block_k, group_m in _candidate_block_sizes(vecs.device, m=M, n=None, k=K):
        try:
            inc.zero_()
            batched_progressive_counts_kernel[grid](
                vecs,
                seq_lens_i32,
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

        out = torch.zeros((B, M, T), device=vecs.device, dtype=torch.int64)
        seq_lens_cpu = seq_lens_i32.to("cpu")
        for b in range(B):
            m_b = int(seq_lens_cpu[b].item())
            if m_b <= 0:
                continue
            out[b, :m_b, :] = torch_impl.progressive_correlation_counts(vecs[b, :m_b, :], epsilons)
        return out

    # Convert incremental unordered counts into prefix ordered-pair counts.
    counts = inc.cumsum(dim=1) * 2
    valid = torch.arange(M, device=vecs.device)[None, :] < seq_lens_i32[:, None]
    return counts * valid.unsqueeze(-1).to(counts.dtype)



def _batched_correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    *,
    seq_lens: torch.Tensor | None = None,
) -> torch.LongTensor:
    """Counts, per batch, the number of pairwise distances <= epsilon for a set of vectors."""
    assert vecs.dim() == 3, "Input must be (B, M, K)"

    B, M, K = vecs.shape
    T = epsilons.shape[0]
    log_epsilons = epsilons.log()
    seq_lens_i32 = _normalize_seq_lens(seq_lens, batch_size=B, max_len=M, device=vecs.device, name="seq_lens")
    counts = torch.zeros((B, T), device=vecs.device, dtype=torch.int64)

    grid = lambda META: (
        B * triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(M, META['BLOCK_SIZE_N']),
    )
    torch.cuda.set_device(vecs.device)
    for block_m, block_n, block_k, group_m in _candidate_block_sizes(vecs.device, m=M, n=M, k=K):
        try:
            counts.zero_()
            batched_correlation_counts_cross_kernel[grid](
                vecs, vecs, seq_lens_i32, seq_lens_i32, log_epsilons, counts,
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

        out = torch.zeros((B, T), device=vecs.device, dtype=torch.int64)
        seq_lens_cpu = seq_lens_i32.to("cpu")
        for b in range(B):
            m_b = int(seq_lens_cpu[b].item())
            if m_b <= 1:
                continue
            out[b, :] = torch_impl.correlation_counts(vecs[b, :m_b, :], epsilons)
        return out
    return counts

def _batched_correlation_counts_cross(
    vecs1: torch.FloatTensor,
    vecs2: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    *,
    seq_lens1: torch.Tensor | None = None,
    seq_lens2: torch.Tensor | None = None,
) -> torch.LongTensor:
    """Counts, per batch, the number of cross-distances <= epsilon between vecs1 and vecs2."""
    if vecs1 is vecs2:
        seq_lens = seq_lens1 if seq_lens1 is not None else seq_lens2
        if seq_lens1 is not None and seq_lens2 is not None:
            l1 = seq_lens1 if isinstance(seq_lens1, torch.Tensor) else torch.as_tensor(seq_lens1)
            l2 = seq_lens2 if isinstance(seq_lens2, torch.Tensor) else torch.as_tensor(seq_lens2)
            if l1.shape != l2.shape or not torch.equal(l1.to("cpu"), l2.to("cpu")):
                raise ValueError("When vecs1 is vecs2, seq_lens and seq_lens_other must match.")
        return _batched_correlation_counts(vecs1, epsilons, seq_lens=seq_lens)

    assert vecs1.dim() == 3 and vecs2.dim() == 3, "Inputs must be (B, M, K) and (B, N, K)"

    B1, M, K1 = vecs1.shape
    B2, N, K2 = vecs2.shape
    assert B1 == B2, "Batch sizes must match"
    assert K1 == K2, "Vector dimensions must match"

    T = epsilons.shape[0]
    seq_lens1_i32 = _normalize_seq_lens(seq_lens1, batch_size=B1, max_len=M, device=vecs1.device, name="seq_lens")
    seq_lens2_i32 = _normalize_seq_lens(seq_lens2, batch_size=B2, max_len=N, device=vecs2.device, name="seq_lens_other")
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
                vecs1, vecs2, seq_lens1_i32, seq_lens2_i32, log_epsilons, counts,
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

        out = torch.zeros((B1, T), device=vecs1.device, dtype=torch.int64)
        lens1_cpu = seq_lens1_i32.to("cpu")
        lens2_cpu = seq_lens2_i32.to("cpu")
        for b in range(B1):
            m_b = int(lens1_cpu[b].item())
            n_b = int(lens2_cpu[b].item())
            if m_b <= 0 or n_b <= 0:
                continue
            out[b, :] = torch_impl.correlation_counts(vecs1[b, :m_b, :], epsilons, vecs_other=vecs2[b, :n_b, :])
        return out
    return counts


def correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    vecs_other: torch.FloatTensor = None,
    seq_lens: torch.Tensor = None,
    seq_lens_other: torch.Tensor = None,
    **kwargs,
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
    if kwargs:
        # Keep parity with other backends: ignore unknown kwargs silently.
        pass
    if vecs_other is None:
        if seq_lens_other is not None:
            raise ValueError("seq_lens_other is only valid when vecs_other is provided.")
        if ndim == 2:
            vecs = vecs.unsqueeze(0)
            if seq_lens is not None:
                raise ValueError("seq_lens is only valid for batched input (B, M, K).")
        counts = _batched_correlation_counts(vecs, epsilons, seq_lens=seq_lens)
        if ndim == 2:
            counts = counts.squeeze(0)
        return counts

    assert vecs.dim() == vecs_other.dim()
    if ndim == 2:
        vecs = vecs.unsqueeze(0)
        vecs_other = vecs_other.unsqueeze(0)
        if seq_lens is not None or seq_lens_other is not None:
            raise ValueError("seq_lens/seq_lens_other are only valid for batched input (B, M, K)/(B, N, K).")
    counts = _batched_correlation_counts_cross(
        vecs,
        vecs_other,
        epsilons,
        seq_lens1=seq_lens,
        seq_lens2=seq_lens_other,
    )
    if ndim == 2:
        counts = counts.squeeze(0)

    return counts


def correlation_integral(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    vecs_other: torch.FloatTensor = None,
    seq_lens: torch.Tensor = None,
    seq_lens_other: torch.Tensor = None,
    **kwargs,
) -> torch.FloatTensor:
    ndim = vecs.dim()
    assert ndim in [2, 3], "Input must be (M, K) or (B, M, K)"
    if vecs_other is None:
        counts = correlation_counts(vecs, epsilons, seq_lens=seq_lens, **kwargs)
        if ndim == 2:
            M = vecs.shape[-2]
            return counts.to(torch.float32) / (M * (M - 1))
        B, M = vecs.shape[0], vecs.shape[1]
        lens = _normalize_seq_lens(seq_lens, batch_size=B, max_len=M, device=vecs.device, name="seq_lens")
        denom = (lens * (lens - 1)).clamp_min(1).to(torch.float32)
        return counts.to(torch.float32) / denom.unsqueeze(-1)
    else:
        counts = correlation_counts(
            vecs,
            epsilons,
            vecs_other,
            seq_lens=seq_lens,
            seq_lens_other=seq_lens_other,
            **kwargs,
        )
        if ndim == 2:
            M, N = vecs.shape[-2], vecs_other.shape[-2]
            return counts.to(torch.float32) / (M * N)
        B, M = vecs.shape[0], vecs.shape[1]
        _, N = vecs_other.shape[0], vecs_other.shape[1]
        lens_m = _normalize_seq_lens(seq_lens, batch_size=B, max_len=M, device=vecs.device, name="seq_lens")
        lens_n = _normalize_seq_lens(seq_lens_other, batch_size=B, max_len=N, device=vecs.device, name="seq_lens_other")
        denom = (lens_m * lens_n).clamp_min(1).to(torch.float32)
        return counts.to(torch.float32) / denom.unsqueeze(-1)


def progressive_correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    seq_lens: torch.Tensor = None,
    **kwargs,
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
        if seq_lens is not None:
            raise ValueError("seq_lens is only valid for batched input (B, M, K).")
        vecs = vecs.unsqueeze(0)
    counts = _batched_progressive_correlation_counts(vecs, epsilons, seq_lens=seq_lens)
    if ndim == 2:
        counts = counts.squeeze(0)
    return counts

def progressive_correlation_integral(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    seq_lens: torch.Tensor = None,
    **kwargs,
) -> torch.FloatTensor:
    """Compute the progressive correlation integral of V[:t] at every step t of the vector sequence V.
    In implementation, the integral is updated at every step t by adding
    an additional integral of V[t] with every vector in V[:t].
    This function supports batched computation.

    vecs: (M, K) or (B, M, K)
    epsilons: (T,)
    """
    counts = progressive_correlation_counts(vecs, epsilons, seq_lens=seq_lens, **kwargs)
    # (B, M, T)
    M = vecs.shape[-2]
    pairs = (
        torch.arange(0, M, device=vecs.device, dtype=torch.float32)
        * torch.arange(1, M + 1, device=vecs.device, dtype=torch.float32)
    )
    pairs[0] = 1e-6
    out = counts.to(torch.float32) / pairs.unsqueeze(-1)
    if vecs.dim() == 3:
        B = vecs.shape[0]
        lens = _normalize_seq_lens(seq_lens, batch_size=B, max_len=M, device=vecs.device, name="seq_lens")
        valid = torch.arange(M, device=vecs.device)[None, :] < lens[:, None]
        out = out * valid.unsqueeze(-1).to(out.dtype)
    return out
