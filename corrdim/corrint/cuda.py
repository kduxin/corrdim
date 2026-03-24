from __future__ import annotations

from typing import Optional, Tuple
import warnings

import torch

from .cuda_extension import load_extension

__all__ = [
    "correlation_counts",
    "correlation_integral",
    "progressive_correlation_counts",
    "progressive_correlation_integral",
]

_FALLBACK_WARNED = False


def _warn_cuda_fallback(reason: Exception | str) -> None:
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    reason_text = str(reason).strip().replace("\n", " ")
    if len(reason_text) > 240:
        reason_text = reason_text[:240] + "..."
    warnings.warn(
        "corrdim.corrint.cuda backend fell back to pytorch backend. "
        f"Reason: {reason_text}",
        RuntimeWarning,
        stacklevel=2,
    )


def _normalize_inputs(
    vecs: torch.Tensor,
    vecs_other: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
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


def _fallback_correlation_counts(
    vecs: torch.Tensor,
    epsilons: torch.Tensor,
    vecs_other: Optional[torch.Tensor],
    kwargs: dict,
    *,
    reason: Exception | str,
) -> torch.Tensor:
    from . import pytorch as torch_impl

    _warn_cuda_fallback(reason)
    return torch_impl.correlation_counts(
        vecs,
        epsilons,
        vecs_other=vecs_other,
        block_size=int(kwargs.get("block_size", 512)),
        show_progress=bool(kwargs.get("show_progress", False)),
        fast=True,
    )


def correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    vecs_other: Optional[torch.FloatTensor] = None,
    **kwargs,
) -> torch.LongTensor:
    vecs_b, other_b, squeezed = _normalize_inputs(vecs, vecs_other)
    if vecs_b.device.type != "cuda":
        return _fallback_correlation_counts(
            vecs,
            epsilons,
            vecs_other,
            kwargs,
            reason="input tensors are not on CUDA device",
        )

    try:
        ext = load_extension()
        log_eps = epsilons.to(device=vecs_b.device, dtype=torch.float32).log().contiguous()
        if other_b is None:
            counts = ext.correlation_counts_self(vecs_b.contiguous(), log_eps)
        else:
            assert vecs_b.shape[0] == other_b.shape[0], "Batch sizes must match"
            assert vecs_b.shape[2] == other_b.shape[2], "Vector dimensions must match"
            counts = ext.correlation_counts_cross(vecs_b.contiguous(), other_b.contiguous(), log_eps)
        torch.cuda.synchronize(device=vecs_b.device)
        return counts.squeeze(0) if squeezed else counts
    except Exception as exc:
        return _fallback_correlation_counts(
            vecs,
            epsilons,
            vecs_other,
            kwargs,
            reason=exc,
        )


def correlation_integral(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    vecs_other: Optional[torch.FloatTensor] = None,
    **kwargs,
) -> torch.FloatTensor:
    counts = correlation_counts(vecs, epsilons, vecs_other=vecs_other, **kwargs)
    if vecs_other is None:
        m = vecs.shape[-2]
        return counts.to(torch.float32) / (m * (m - 1))
    m, n = vecs.shape[-2], vecs_other.shape[-2]
    return counts.to(torch.float32) / (m * n)


def progressive_correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    **kwargs,
) -> torch.LongTensor:
    vecs_b, _other_b, squeezed = _normalize_inputs(vecs, None)

    if vecs_b.device.type != "cuda":
        from . import pytorch as torch_impl

        _warn_cuda_fallback("input tensors are not on CUDA device")
        return torch_impl.progressive_correlation_counts(
            vecs,
            epsilons,
            block_size=int(kwargs.get("block_size", 512)),
            show_progress=bool(kwargs.get("show_progress", False)),
            fast=True,
        )

    try:
        ext = load_extension()
        log_eps = epsilons.to(device=vecs_b.device, dtype=torch.float32).log().contiguous()
        inc = ext.progressive_counts_self(vecs_b.contiguous(), log_eps)
        out = inc.cumsum(dim=1) * 2
        torch.cuda.synchronize(device=vecs_b.device)
        return out.squeeze(0) if squeezed else out
    except Exception as exc:
        from . import pytorch as torch_impl

        _warn_cuda_fallback(exc)
        return torch_impl.progressive_correlation_counts(
            vecs,
            epsilons,
            block_size=int(kwargs.get("block_size", 512)),
            show_progress=bool(kwargs.get("show_progress", False)),
            fast=True,
        )


def progressive_correlation_integral(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    **kwargs,
) -> torch.FloatTensor:
    counts = progressive_correlation_counts(vecs, epsilons, **kwargs)
    m = vecs.shape[-2]
    pairs = (
        torch.arange(0, m, device=vecs.device, dtype=torch.float32)
        * torch.arange(1, m + 1, device=vecs.device, dtype=torch.float32)
    )
    pairs[0] = 1e-6
    if vecs.dim() == 2:
        return counts.to(torch.float32) / pairs[:, None]
    return counts.to(torch.float32) / pairs[None, :, None]

