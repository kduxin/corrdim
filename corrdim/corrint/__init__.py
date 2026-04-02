from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, Optional, Union

import torch


class CorrIntBackend(str, Enum):
    """Correlation integral backend selector."""

    AUTO = "auto"
    CUDA = "cuda"
    TRITON = "triton"
    PYTORCH = "pytorch"
    PYTORCH_FAST = "pytorch_fast"


BackendLike = Union[str, CorrIntBackend, None]

_KNOWN_BACKENDS = {"auto", "cuda", "triton", "pytorch", "pytorch_fast"}


def _normalize_backend_name(name: Any, *, source: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in _KNOWN_BACKENDS:
        raise ValueError(
            f"Unknown backend from {source}: {name!r}. "
            f"Available: {sorted(_KNOWN_BACKENDS)}"
        )
    return normalized


def _env_default_backend() -> str:
    # Canonical env var for selecting the correlation-integral backend.
    raw = os.environ.get("CORRDIM_CORRINT_BACKEND")
    source = "CORRDIM_CORRINT_BACKEND"
    if raw is None:
        return "triton"
    return _normalize_backend_name(raw, source=source)


# Default backend: environment variable first, otherwise auto (auto-detect).
_DEFAULT_BACKEND: str = _env_default_backend()


def _normalize_backend(backend: BackendLike) -> str:
    if backend is None:
        name = _DEFAULT_BACKEND
    elif isinstance(backend, CorrIntBackend):
        name = backend.value
    else:
        name = str(backend)
    return _normalize_backend_name(name, source="set_corrint_backend/backend")


def _triton_is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from . import triton as _impl  # noqa: F401
    except Exception as e:  # pragma: no cover
        return False
    return True


def _cuda_is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from . import cuda as _impl  # noqa: F401
    except Exception:  # pragma: no cover
        return False
    return True


def resolve_corrint_backend(backend: BackendLike = None) -> str:
    """Resolve AUTO to a concrete backend name."""
    name = _normalize_backend(backend)
    if name != "auto":
        return name
    return "cuda" if _cuda_is_available() else "pytorch"


def available_corrint_backends() -> Dict[str, bool]:
    """Return availability of known backends."""
    return {
        "cuda": _cuda_is_available(),
        "triton": _triton_is_available(),
        "pytorch": True,
        "pytorch_fast": True,
    }


def set_corrint_backend(backend: BackendLike = "auto") -> str:
    """Set process-wide default backend; returns the resolved backend name."""
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = _normalize_backend(backend)
    return resolve_corrint_backend(None)


def get_corrint_backend() -> str:
    """Get the current default backend (AUTO is resolved)."""
    return resolve_corrint_backend(None)


def _all_cuda(*tensors: Optional[torch.Tensor]) -> bool:
    present = [t for t in tensors if t is not None]
    return len(present) > 0 and all(getattr(t, "device", None) is not None and t.device.type == "cuda" for t in present)


def _select_impl(backend: BackendLike, *tensors: Optional[torch.Tensor]):
    name = resolve_corrint_backend(backend)
    if name == "cuda" and _all_cuda(*tensors):
        from . import cuda as impl

        return name, impl, {}
    if name == "triton" and _all_cuda(*tensors):
        from . import triton as impl

        return name, impl, {}
    from . import pytorch as impl

    return name, impl, {"fast": name == "pytorch_fast"}


def correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    vecs_other: Optional[torch.FloatTensor] = None,
    *,
    backend: BackendLike = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Return counts (unnormalized); implemented by the selected backend module."""
    name, impl, extra = _select_impl(backend, vecs, vecs_other)
    if vecs_other is None:
        return impl.correlation_counts(vecs, epsilons, **extra, **kwargs)
    return impl.correlation_counts(vecs, epsilons, vecs_other=vecs_other, **extra, **kwargs)


def correlation_integral(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    vecs_other: Optional[torch.FloatTensor] = None,
    return_counts: bool = False,
    *,
    backend: BackendLike = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Return correlation integral; if return_counts=True, return counts instead."""
    if return_counts:
        return correlation_counts(vecs, epsilons, vecs_other=vecs_other, backend=backend, **kwargs)

    name, impl, extra = _select_impl(backend, vecs, vecs_other)
    if vecs_other is None:
        return impl.correlation_integral(vecs, epsilons, **extra, **kwargs)
    return impl.correlation_integral(vecs, epsilons, vecs_other=vecs_other, **extra, **kwargs)


def progressive_correlation_counts(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    *,
    backend: BackendLike = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Progressive counts over sequence prefixes (if implemented by backend)."""
    name, impl, extra = _select_impl(backend, vecs)
    fn = getattr(impl, "progressive_correlation_counts", None)
    if fn is None:
        raise NotImplementedError(f"backend={name!r} does not implement progressive_correlation_counts")
    return fn(vecs, epsilons, **extra, **kwargs)


def progressive_correlation_integral(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    return_counts: bool = False,
    *,
    backend: BackendLike = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Progressive correlation integral over sequence prefixes (if implemented by backend)."""
    if return_counts:
        return progressive_correlation_counts(vecs, epsilons, backend=backend, **kwargs)
    name, impl, extra = _select_impl(backend, vecs)
    fn = getattr(impl, "progressive_correlation_integral", None)
    if fn is None:
        raise NotImplementedError(f"backend={name!r} does not implement progressive_correlation_integral")
    return fn(vecs, epsilons, **extra, **kwargs)


def correlation_integral_cross(
    vecs1: torch.FloatTensor,
    vecs2: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    *,
    backend: BackendLike = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Cross correlation integral (requires backend support for vecs_other)."""
    return correlation_integral(vecs1, epsilons, vecs_other=vecs2, backend=backend, **kwargs)


def correlation_integral_sequence(
    vecs: torch.FloatTensor,
    epsilons: torch.FloatTensor,
    *,
    backend: BackendLike = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Correlation integral curve for every prefix (same as progressive_correlation_integral)."""
    return progressive_correlation_integral(vecs, epsilons, backend=backend, **kwargs)


__all__ = [
    "CorrIntBackend",
    "available_corrint_backends",
    "resolve_corrint_backend",
    "set_corrint_backend",
    "get_corrint_backend",
    "correlation_counts",
    "correlation_integral",
    "progressive_correlation_counts",
    "progressive_correlation_integral",
]