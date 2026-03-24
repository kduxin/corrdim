from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Optional, Tuple, Union

import numpy as np
import torch

from .corrint import correlation_integral, progressive_correlation_integral
from .models import LanguageModelWrapper, create_model_wrapper
from .types import CurveResult, ProgressiveCurveResult
from .utils import clamp

ModelLike = Union[str, LanguageModelWrapper]
_MODEL_CACHE_MAX_SIZE = 2
_MODEL_CACHE: "OrderedDict[tuple, LanguageModelWrapper]" = OrderedDict()
_MODEL_CACHE_LOCK = Lock()


def _freeze_kwargs(kwargs: dict) -> tuple:
    return tuple(sorted((str(key), repr(value)) for key, value in kwargs.items()))


def _cache_key(model_name: str, tokenizer: Optional[object], device: Optional[str], kwargs: dict) -> tuple:
    tokenizer_key = None if tokenizer is None else id(tokenizer)
    return (model_name, tokenizer_key, device, _freeze_kwargs(kwargs))


def clear_model_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _resolve_model_wrapper(
    model: ModelLike,
    tokenizer: Optional[object] = None,
    device: Optional[str] = None,
    **kwargs,
) -> LanguageModelWrapper:
    if isinstance(model, str):
        key = _cache_key(model, tokenizer, device, kwargs)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                _MODEL_CACHE.move_to_end(key)
                return cached
        wrapper = create_model_wrapper(model, tokenizer=tokenizer, device=device, **kwargs)
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[key] = wrapper
            _MODEL_CACHE.move_to_end(key)
            while len(_MODEL_CACHE) > _MODEL_CACHE_MAX_SIZE:
                _MODEL_CACHE.popitem(last=False)
        return wrapper
    return model


def _make_epsilons(vecs: torch.Tensor, epsilon_range: Tuple[float, float], num_epsilon: int) -> torch.Tensor:
    if not torch.isfinite(vecs).all():
        raise ValueError("Found nan or inf in vectors.")
    if vecs.shape[-2] <= 100:
        raise ValueError(f"The sequence length is too short ({vecs.shape[-2]} tokens). Please use at least 100 tokens.")
    return torch.logspace(
        np.log10(float(epsilon_range[0])),
        np.log10(float(epsilon_range[1])),
        num_epsilon,
        device=vecs.device,
    )


def curve_from_vectors(
    vectors: torch.Tensor,
    epsilon_range: Tuple[float, float] = (10**-20.0, 10**20.0),
    num_epsilon: int = 1024,
    block_size: int = 512,
    show_progress: bool = False,
    backend: Optional[str] = None,
) -> CurveResult:
    epsilons = _make_epsilons(vectors, epsilon_range, num_epsilon)
    corrints = correlation_integral(vectors, epsilons, block_size=block_size, show_progress=show_progress, backend=backend)
    eps_clamped, corr_clamped = clamp(epsilons, corrints, low=float(corrints.min()), high=0.95)
    if len(eps_clamped) >= 2:
        epsilons, corrints = eps_clamped, corr_clamped
    return CurveResult(
        sequence_length=int(vectors.shape[-2]),
        epsilons=epsilons.detach().cpu().numpy(),
        corrints=corrints.detach().cpu().numpy(),
    )


def curve_from_vectors_batch(
    vectors_batch: torch.Tensor,
    epsilon_range: Tuple[float, float] = (10**-20.0, 10**20.0),
    num_epsilon: int = 1024,
    block_size: int = 512,
    show_progress: bool = False,
    backend: Optional[str] = None,
) -> list[CurveResult]:
    if vectors_batch.dim() != 3:
        raise ValueError("vectors_batch must have shape (B, M, K).")
    return [
        curve_from_vectors(
            vectors_batch[idx],
            epsilon_range=epsilon_range,
            num_epsilon=num_epsilon,
            block_size=block_size,
            show_progress=show_progress,
            backend=backend,
        )
        for idx in range(vectors_batch.shape[0])
    ]


def progressive_curve_from_vectors(
    vectors: torch.Tensor,
    epsilon_range: Tuple[float, float] = (10**-20.0, 10**20.0),
    num_epsilon: int = 1024,
    block_size: int = 512,
    show_progress: bool = False,
    backend: Optional[str] = None,
) -> ProgressiveCurveResult:
    epsilons = _make_epsilons(vectors, epsilon_range, num_epsilon)
    corrints_progressive = progressive_correlation_integral(
        vectors,
        epsilons,
        block_size=block_size,
        show_progress=show_progress,
        backend=backend,
    )
    return ProgressiveCurveResult(
        sequence_length=int(vectors.shape[-2]),
        epsilons=epsilons.detach().cpu().numpy(),
        corrints_progressive=corrints_progressive.detach().cpu().numpy(),
    )


def progressive_curve_from_vectors_batch(
    vectors_batch: torch.Tensor,
    epsilon_range: Tuple[float, float] = (10**-20.0, 10**20.0),
    num_epsilon: int = 1024,
    block_size: int = 512,
    show_progress: bool = False,
    backend: Optional[str] = None,
) -> list[ProgressiveCurveResult]:
    if vectors_batch.dim() != 3:
        raise ValueError("vectors_batch must have shape (B, M, K).")
    return [
        progressive_curve_from_vectors(
            vectors_batch[idx],
            epsilon_range=epsilon_range,
            num_epsilon=num_epsilon,
            block_size=block_size,
            show_progress=show_progress,
            backend=backend,
        )
        for idx in range(vectors_batch.shape[0])
    ]


def _text_to_vectors(
    model_wrapper: LanguageModelWrapper,
    text: str,
    dim_reduction: Optional[int] = 8192,
    context_length: Optional[int] = None,
    stride: int = 1,
    show_progress: bool = False,
    precision: torch.dtype = torch.float32,
) -> torch.Tensor:
    return model_wrapper.get_log_probabilities(
        text,
        dim_reduction=dim_reduction,
        context_length=context_length,
        stride=stride,
        show_progress=show_progress,
    ).type(precision)


def curve_from_text(
    text: str,
    model: ModelLike,
    tokenizer: Optional[object] = None,
    context_length: Optional[int] = None,
    dim_reduction: Optional[int] = 8192,
    stride: int = 1,
    epsilon_range: Tuple[float, float] = (10**-20.0, 10**20.0),
    num_epsilon: int = 1024,
    block_size: int = 512,
    show_progress: bool = False,
    precision: torch.dtype = torch.float32,
    backend: Optional[str] = None,
    **model_kwargs,
) -> CurveResult:
    model_wrapper = _resolve_model_wrapper(model, tokenizer=tokenizer, **model_kwargs)
    vectors = _text_to_vectors(
        model_wrapper,
        text=text,
        dim_reduction=dim_reduction,
        context_length=context_length,
        stride=stride,
        show_progress=show_progress,
        precision=precision,
    )
    return curve_from_vectors(
        vectors=vectors,
        epsilon_range=epsilon_range,
        num_epsilon=num_epsilon,
        block_size=block_size,
        show_progress=show_progress,
        backend=backend,
    )


def curve_from_texts(
    texts: list[str],
    model: ModelLike,
    tokenizer: Optional[object] = None,
    context_length: Optional[int] = None,
    dim_reduction: Optional[int] = 8192,
    stride: int = 1,
    epsilon_range: Tuple[float, float] = (10**-20.0, 10**20.0),
    num_epsilon: int = 1024,
    block_size: int = 512,
    show_progress: bool = False,
    precision: torch.dtype = torch.float32,
    backend: Optional[str] = None,
    **model_kwargs,
) -> list[CurveResult]:
    model_wrapper = _resolve_model_wrapper(model, tokenizer=tokenizer, **model_kwargs)
    return [
        curve_from_text(
            text,
            model=model_wrapper,
            context_length=context_length,
            dim_reduction=dim_reduction,
            stride=stride,
            epsilon_range=epsilon_range,
            num_epsilon=num_epsilon,
            block_size=block_size,
            show_progress=show_progress,
            precision=precision,
            backend=backend,
        )
        for text in texts
    ]


def progressive_curve_from_text(
    text: str,
    model: ModelLike,
    tokenizer: Optional[object] = None,
    context_length: Optional[int] = None,
    dim_reduction: Optional[int] = 8192,
    stride: int = 1,
    epsilon_range: Tuple[float, float] = (10**-20.0, 10**20.0),
    num_epsilon: int = 1024,
    block_size: int = 512,
    show_progress: bool = False,
    precision: torch.dtype = torch.float32,
    backend: Optional[str] = None,
    **model_kwargs,
) -> ProgressiveCurveResult:
    model_wrapper = _resolve_model_wrapper(model, tokenizer=tokenizer, **model_kwargs)
    vectors = _text_to_vectors(
        model_wrapper,
        text=text,
        dim_reduction=dim_reduction,
        context_length=context_length,
        stride=stride,
        show_progress=show_progress,
        precision=precision,
    )
    return progressive_curve_from_vectors(
        vectors=vectors,
        epsilon_range=epsilon_range,
        num_epsilon=num_epsilon,
        block_size=block_size,
        show_progress=show_progress,
        backend=backend,
    )


def progressive_curve_from_texts(
    texts: list[str],
    model: ModelLike,
    tokenizer: Optional[object] = None,
    context_length: Optional[int] = None,
    dim_reduction: Optional[int] = 8192,
    stride: int = 1,
    epsilon_range: Tuple[float, float] = (10**-20.0, 10**20.0),
    num_epsilon: int = 1024,
    block_size: int = 512,
    show_progress: bool = False,
    precision: torch.dtype = torch.float32,
    backend: Optional[str] = None,
    **model_kwargs,
) -> list[ProgressiveCurveResult]:
    model_wrapper = _resolve_model_wrapper(model, tokenizer=tokenizer, **model_kwargs)
    return [
        progressive_curve_from_text(
            text,
            model=model_wrapper,
            context_length=context_length,
            dim_reduction=dim_reduction,
            stride=stride,
            epsilon_range=epsilon_range,
            num_epsilon=num_epsilon,
            block_size=block_size,
            show_progress=show_progress,
            precision=precision,
            backend=backend,
        )
        for text in texts
    ]
