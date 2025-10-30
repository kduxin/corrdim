import torch
import triton
import triton.language as tl

@triton.jit
def correlation_integral_kernel(
        # Pointers to matrices
        vecs_ptr, logthreshs_ptr, counts_ptr,
        # Matrix dimensions
        M, K, T: tl.constexpr,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_vm, stride_vk,  #
        stride_t,
        stride_c,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Compute correlation integral counts for upper-triangular blocks.
    Each program handles a (BLOCK_SIZE_M x BLOCK_SIZE_M) block and
    accumulates pairwise L2 distances in log-space against thresholds.
    """
    # Map program id to grouped (pid_m, pid_n); process only upper triangle
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Avoid blocks below the diagonal
    if pid_m > pid_n:
        return

    # Basic integer bound assumptions for better codegen
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_vm > 0)
    tl.assume(stride_vk > 0)
    tl.assume(stride_t > 0)
    tl.assume(stride_c > 0)

    # Create base pointers for current M-block rows and advance along K
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = vecs_ptr + (offs_am[:, None] * stride_vm + offs_k[None, :] * stride_vk)
    b_ptrs = vecs_ptr + (offs_bn[:, None] * stride_vm + offs_k[None, :] * stride_vk)
    offs_am_mask = offs_am < M
    offs_bn_mask = offs_bn < M

    # Accumulate squared distances over K tiles
    distsq = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offs < K
        
        a = tl.load(a_ptrs, mask=offs_am_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=offs_bn_mask[:, None] & k_mask[None, :], other=0.0)

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        ab = tl.dot(a, b.T)
        a2 = tl.sum(a * a, axis=1)
        b2 = tl.sum(b * b, axis=1)
        distsq += a2[:, None] + b2[None, :] - 2 * ab

        a_ptrs += BLOCK_SIZE_K * stride_vk
        b_ptrs += BLOCK_SIZE_K * stride_vk

    # Mask out invalid rows/cols and strictly keep i < j on diagonal blocks
    mask = offs_am_mask[:, None] & offs_bn_mask[None, :]
    if pid_m == pid_n:
        triu_mask = tl.arange(0, BLOCK_SIZE_M)[:, None] < tl.arange(0, BLOCK_SIZE_M)[None, :]
        mask = mask & triu_mask
    distsq = tl.where(mask, distsq, torch.inf)
    # Convert to log distances (safe for zero by mapping to -inf)
    logdist = tl.where(distsq > 0, 0.5 * tl.log(distsq), -torch.inf)

    for i in range(0, T):
        logthresh = tl.load(logthreshs_ptr + i * stride_t)
        count = tl.sum((logdist < logthresh).to(tl.int64))
        tl.atomic_add(counts_ptr + i * stride_c, count.to(tl.int64))


def correlation_integral(vecs: torch.FloatTensor, epsilons: torch.FloatTensor) -> torch.LongTensor:
    # Check constraints.
    assert vecs.is_contiguous(), "Matrix A must be contiguous"
    M, K = vecs.shape
    T = epsilons.shape[0]
    log_epsilons = epsilons.log()
    counts = torch.zeros(T, device=vecs.device, dtype=torch.int64)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(M, META['BLOCK_SIZE_M']), )
    torch.cuda.set_device(vecs.device)
    correlation_integral_kernel[grid](
        vecs, log_epsilons, counts, 
        M, K, T, 
        vecs.stride(0),
        vecs.stride(1), 
        log_epsilons.stride(0), 
        counts.stride(0), 
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
    )
    torch.cuda.synchronize()
    corrints = counts.to(torch.float64) / (M * (M - 1) / 2)
    return corrints