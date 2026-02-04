from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, Optional, Union

import torch


class CorrIntBackend(str, Enum):
    """Correlation integral backend selector."""

    AUTO = "auto"
    TRITON = "triton"
    PYTORCH = "pytorch"
    PYTORCH_FAST = "pytorch_fast"


BackendLike = Union[str, CorrIntBackend, None]

_KNOWN_BACKENDS = {"auto", "triton", "pytorch", "pytorch_fast"}


def _env_default_backend() -> str:
    # Unified env var name: CORRDIM_BACKEND
    return (
        (os.environ.get("CORRDIM_BACKEND") or "auto")
        .strip()
        .lower()
        or "auto"
    )


# Default backend: environment variable first, otherwise auto (auto-detect).
_DEFAULT_BACKEND: str = _env_default_backend()


def _normalize_backend(backend: BackendLike) -> str:
    if backend is None:
        name = _DEFAULT_BACKEND
    elif isinstance(backend, CorrIntBackend):
        name = backend.value
    else:
        name = str(backend)
    name = name.strip().lower()
    if name not in _KNOWN_BACKENDS:
        raise ValueError(
            f"Unknown backend={backend!r}. Available: {sorted(_KNOWN_BACKENDS)}"
        )
    return name


def _triton_is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        from . import triton as _impl  # noqa: F401
    except Exception as e:  # pragma: no cover
        return False
    return True


def resolve_corrint_backend(backend: BackendLike = None) -> str:
    """Resolve AUTO to a concrete backend name."""
    name = _normalize_backend(backend)
    if name != "auto":
        return name
    return "triton" if _triton_is_available() else "pytorch"


def available_corrint_backends() -> Dict[str, bool]:
    """Return availability of known backends."""
    return {"triton": _triton_is_available(), "pytorch": True, "pytorch_fast": True}


def set_corrint_backend(backend: BackendLike = "auto") -> str:
    """Set process-wide default backend; returns the resolved backend name."""
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = _normalize_backend(backend)
    return resolve_corrint_backend(None)


def get_corrint_backend() -> str:
    """Get the current default backend (AUTO is resolved)."""
    return resolve_corrint_backend(None)


def _select_impl(backend: BackendLike):
    name = resolve_corrint_backend(backend)
    if name == "triton":
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
    name, impl, extra = _select_impl(backend)
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

    name, impl, extra = _select_impl(backend)
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
    name, impl, extra = _select_impl(backend)
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
    name, impl, extra = _select_impl(backend)
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