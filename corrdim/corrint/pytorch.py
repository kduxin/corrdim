from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

try:
    import tqdm
except Exception:  # pragma: no cover
    tqdm = None

__all__ = [
    "correlation_counts",
    "correlation_integral",
    "progressive_correlation_counts",
    "progressive_correlation_integral",
]


def _tqdm_range(*args, show_progress: bool, desc: str):
    if (not show_progress) or tqdm is None:
        return range(*args)
    return tqdm.trange(*args, disable=not show_progress, desc=desc)


def _cdist(a: torch.Tensor, b: torch.Tensor, *, fast: bool) -> torch.Tensor:
    if fast:
        return torch.cdist(a, b, p=2, compute_mode="use_mm_for_euclid_dist_if_necessary")
    return torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")


def _assert_finite(distances: torch.Tensor) -> None:
    assert not (torch.isnan(distances).any() or torch.isinf(distances).any()), (
        "Found nan or inf in distances. Consider using higher precision for distance computation."
    )


def _counts_self_2d(
    vecs: torch.Tensor,
    epsilons: torch.Tensor,
    *,
    show_progress: bool,
    block_size: int,
    fast: bool,
) -> torch.Tensor:
    """Return counts for vecs vs vecs.

    Semantics match the Triton backend: (i, j) and (j, i) are counted twice; diagonal is excluded.
    For epsilon=+inf, the count equals M * (M - 1).
    """
    m = vecs.shape[0]
    eps_log = epsilons.log()
    counts = torch.zeros(epsilons.shape[0], dtype=torch.int64, device=vecs.device)
    for i in _tqdm_range(0, m, block_size, show_progress=show_progress, desc="Computing correlation counts"):
        a = vecs[i : i + block_size].to(torch.float32)
        for j in range(i, m, block_size):
            b = vecs[j : j + block_size].to(torch.float32)
            d = _cdist(a, b, fast=fast)
            _assert_finite(d)
            if i == j:
                # Keep only strictly upper triangle within the diagonal block.
                tril = torch.tril_indices(d.shape[0], d.shape[1], device=d.device)
                d[tril[0], tril[1]] = float("inf")

            d_log = torch.sort(d.reshape(-1), descending=False, stable=False).values.log()
            counts += torch.searchsorted(d_log, eps_log, right=True).to(torch.int64)

    # Convert "upper triangle once" into "ordered pairs" by doubling.
    return counts * 2


def _counts_cross_2d(
    vecs1: torch.Tensor,
    vecs2: torch.Tensor,
    epsilons: torch.Tensor,
    *,
    show_progress: bool,
    block_size: int,
    fast: bool,
) -> torch.Tensor:
    """Return counts for vecs1 vs vecs2.

    Cross semantics: counts all pairs (i, j) with i in vecs1 and j in vecs2 exactly once.
    For epsilon=+inf, the count equals M * N.
    """
    m = vecs1.shape[0]
    n = vecs2.shape[0]
    eps_log = epsilons.log()
    counts = torch.zeros(epsilons.shape[0], dtype=torch.int64, device=vecs1.device)
    for i in _tqdm_range(0, m, block_size, show_progress=show_progress, desc="Computing correlation counts (cross)"):
        a = vecs1[i : i + block_size].to(torch.float32)
        for j in range(0, n, block_size):
            b = vecs2[j : j + block_size].to(torch.float32)
            d = _cdist(a, b, fast=fast)
            _assert_finite(d)

            d_log = torch.sort(d.reshape(-1), descending=False, stable=False).values.log()
            counts += torch.searchsorted(d_log, eps_log, right=True).to(torch.int64)

    return counts


def _normalize_inputs(
    vecs: torch.Tensor,
    vecs_other: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
    """Normalize to (B, M, K) and (B, N, K), return (vecs_b, other_b, squeezed)."""
    ndim = vecs.dim()
    assert ndim in (2, 3), "Input must be (M, K) or (B, M, K)"
    if vecs_other is not None:
        assert vecs_other.dim() == ndim, "vecs_other must have the same ndim as vecs"
    squeezed = False
    if ndim == 2:
        vecs = vecs.unsqueeze(0)
        vecs_other = vecs_other.unsqueeze(0) if vecs_other is not None else None
        squeezed = True
    return vecs, vecs_other, squeezed

def correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    vecs_other: Optional[torch.FloatTensor] = None,
    *,
    show_progress: bool = False,
    block_size: int = 512,
    fast: bool = False,
) -> torch.LongTensor:
    """Compute correlation counts.

    - If vecs_other is None: counts ordered pairs within vecs excluding diagonal (so (i, j) and (j, i) both count).
    - If vecs_other is provided: counts all cross pairs (i in vecs, j in vecs_other) once.
    - Supports inputs of shape (M, K) or (B, M, K).
    """
    vecs_b, other_b, squeezed = _normalize_inputs(vecs, vecs_other)
    bsz = vecs_b.shape[0]

    outs = []
    for b in range(bsz):
        v = vecs_b[b]
        o = other_b[b] if other_b is not None else None
        if o is not None and o is v:
            o = None

        if o is None:
            out = _counts_self_2d(v, epsilons, show_progress=show_progress, block_size=block_size, fast=fast)
        else:
            out = _counts_cross_2d(v, o, epsilons, show_progress=show_progress, block_size=block_size, fast=fast)
        outs.append(out)

    out_b = torch.stack(outs, dim=0)
    return out_b.squeeze(0) if squeezed else out_b

def correlation_integral(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    vecs_other: Optional[torch.FloatTensor] = None,
    *,
    show_progress: bool = False,
    block_size: int = 512,
    fast: bool = False,
) -> torch.FloatTensor:
    """Compute correlation integral (normalized counts)."""
    counts = correlation_counts(
        vecs,
        epsilons,
        vecs_other=vecs_other,
        show_progress=show_progress,
        block_size=block_size,
        fast=fast,
    )

    if vecs_other is None:
        m = vecs.shape[-2]
        denom = float(m * (m - 1))
    else:
        m = vecs.shape[-2]
        n = vecs_other.shape[-2]
        denom = float(m * n)
    return counts.to(torch.float32) / denom


def progressive_correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    *,
    show_progress: bool = False,
    block_size: int = 512,
    fast: bool = False,
) -> torch.LongTensor:
    """Compute per-step incremental cross-counts for sequence prefixes.

    This matches the Triton backend behavior: at step t, counts pairs between vecs[:t] and vecs[t:t+1],
    i.e. compares the current vector against all previous vectors (excluding self).
    """
    vecs_b, _other_b, squeezed = _normalize_inputs(vecs, None)
    bsz, m, _k = vecs_b.shape
    t = epsilons.shape[0]

    out = torch.zeros((bsz, m, t), device=vecs.device, dtype=torch.int64)
    it = range(1, m)
    if show_progress and tqdm is not None:
        it = tqdm.trange(1, m, disable=not show_progress, desc="Computing progressive correlation counts")

    for i in it:
        out[:, i, :] = correlation_counts(
            vecs_b[:, : i, :],
            epsilons,
            vecs_other=vecs_b[:, i : i + 1, :],
            show_progress=False,
            block_size=block_size,
            fast=fast,
        ) * 2
    out = out.cumsum(dim=-2)
    return out.squeeze(0) if squeezed else out


def progressive_correlation_integral(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    *,
    show_progress: bool = False,
    block_size: int = 512,
    fast: bool = False,
) -> torch.FloatTensor:
    """Compute progressive correlation integral from progressive counts."""
    counts = progressive_correlation_counts(
        vecs,
        epsilons,
        show_progress=show_progress,
        block_size=block_size,
        fast=fast,
    )

    if vecs.dim() == 2:
        m = vecs.shape[0]
        pairs = torch.arange(0, m, device=vecs.device, dtype=torch.float32) * torch.arange(1, m + 1, device=vecs.device, dtype=torch.float32)
        pairs[0] = 1.0
        return counts.to(torch.float32) / pairs[:, None]

    m = vecs.shape[1]
    pairs = torch.arange(0, m, device=vecs.device, dtype=torch.float32) * torch.arange(1, m + 1, device=vecs.device, dtype=torch.float32)
    pairs[0] = 1.0
    return counts.to(torch.float32) / pairs[None, :, None]