from __future__ import annotations

from typing import Optional, Tuple

import torch

from .dimension import estimate_dimension_from_curve
from .low_level import ModelLike, curve_from_text, curve_from_texts, progressive_curve_from_text
from .types import CurveResult, DimensionResult

DEFAULT_EPSILON_RANGE: Tuple[float, float] = (10**-20.0, 10**20.0)


def _truncate_text_by_tokens(
    text: str,
    truncation_tokens: int,
    model: ModelLike,
    tokenizer: Optional[object] = None,
) -> str:
    if truncation_tokens < 1:
        raise ValueError("truncation_tokens must be a positive integer.")

    tokenizer_obj = tokenizer
    if tokenizer_obj is None and hasattr(model, "tokenizer"):
        tokenizer_obj = model.tokenizer
    if tokenizer_obj is None and isinstance(model, str):
        from transformers import AutoTokenizer

        tokenizer_obj = AutoTokenizer.from_pretrained(model)

    if tokenizer_obj is None:
        raise ValueError("Cannot truncate by tokens without a tokenizer. Please pass tokenizer explicitly.")

    tokens = tokenizer_obj.encode(text, add_special_tokens=False)
    if len(tokens) <= truncation_tokens:
        return text
    return tokenizer_obj.decode(tokens[:truncation_tokens], skip_special_tokens=False)

def measure_text(
    text: str,
    model: ModelLike,
    tokenizer: Optional[object] = None,
    truncation_tokens: Optional[int] = None,
    context_length: Optional[int] = None,
    dim_reduction: Optional[int] = 8192,
    stride: int = 1,
    correlation_integral_range: Optional[Tuple[float, float]] = None,
    epsilon_range: Optional[Tuple[float, float]] = None,
    num_epsilon: int = 1024,
    block_size: int = 512,
    show_progress: bool = False,
    precision: torch.dtype = torch.float16,
    backend: Optional[str] = None,
    **model_kwargs,
) -> DimensionResult:
    if truncation_tokens is not None:
        text = _truncate_text_by_tokens(
            text=text,
            truncation_tokens=truncation_tokens,
            model=model,
            tokenizer=tokenizer,
        )

    effective_epsilon_range = DEFAULT_EPSILON_RANGE if epsilon_range is None else epsilon_range

    curve = curve_from_text(
        text=text,
        model=model,
        tokenizer=tokenizer,
        context_length=context_length,
        dim_reduction=dim_reduction,
        stride=stride,
        epsilon_range=effective_epsilon_range,
        num_epsilon=num_epsilon,
        block_size=block_size,
        show_progress=show_progress,
        precision=precision,
        backend=backend,
        **model_kwargs,
    )
    return estimate_dimension_from_curve(
        curve,
        correlation_integral_range=correlation_integral_range,
        epsilon_range=epsilon_range,
    )


def measure_text_progressive(
    text: str,
    model: ModelLike,
    tokenizer: Optional[object] = None,
    truncation_tokens: Optional[int] = None,
    skip_prefix_tokens: int = 100,
    measure_every_tokens: int = 100,
    context_length: Optional[int] = None,
    dim_reduction: Optional[int] = 8192,
    stride: int = 1,
    correlation_integral_range: Optional[Tuple[float, float]] = None,
    epsilon_range: Optional[Tuple[float, float]] = None,
    num_epsilon: int = 1024,
    block_size: int = 512,
    show_progress: bool = False,
    precision: torch.dtype = torch.float16,
    backend: Optional[str] = None,
    **model_kwargs,
) -> dict[int, DimensionResult]:
    """Compute progressive curves once, then fit correlation dimension at sampled prefixes.

    For each index ``i`` in ``range(skip_prefix_tokens, sequence_length, measure_every_tokens)``,
    uses row ``corrints_progressive[i]`` with the shared ``epsilons`` grid. Return value maps
    ``i`` to :class:`~corrdim.types.DimensionResult` (same structure as :func:`measure_text`).
    Other arguments follow :func:`measure_text` / :func:`~corrdim.low_level.progressive_curve_from_text`.
    """
    if skip_prefix_tokens < 0:
        raise ValueError("skip_prefix_tokens must be non-negative.")
    if measure_every_tokens < 1:
        raise ValueError("measure_every_tokens must be a positive integer.")

    if truncation_tokens is not None:
        text = _truncate_text_by_tokens(
            text=text,
            truncation_tokens=truncation_tokens,
            model=model,
            tokenizer=tokenizer,
        )

    effective_epsilon_range = DEFAULT_EPSILON_RANGE if epsilon_range is None else epsilon_range

    prog = progressive_curve_from_text(
        text=text,
        model=model,
        tokenizer=tokenizer,
        context_length=context_length,
        dim_reduction=dim_reduction,
        stride=stride,
        epsilon_range=effective_epsilon_range,
        num_epsilon=num_epsilon,
        block_size=block_size,
        show_progress=show_progress,
        precision=precision,
        backend=backend,
        **model_kwargs,
    )

    out: dict[int, DimensionResult] = {}
    for i in range(skip_prefix_tokens, prog.sequence_length, measure_every_tokens):
        curve = CurveResult(
            sequence_length=prog.sequence_length,
            epsilons=prog.epsilons,
            corrints=prog.corrints_progressive[i],
        )
        out[i] = estimate_dimension_from_curve(
            curve,
            correlation_integral_range=correlation_integral_range,
            epsilon_range=epsilon_range,
        )
    return out


def measure_texts(
    texts: list[str],
    model: ModelLike,
    tokenizer: Optional[object] = None,
    context_length: Optional[int] = None,
    dim_reduction: Optional[int] = 8192,
    stride: int = 1,
    correlation_integral_range: Optional[Tuple[float, float]] = None,
    epsilon_range: Optional[Tuple[float, float]] = None,
    num_epsilon: int = 1024,
    block_size: int = 512,
    show_progress: bool = False,
    precision: torch.dtype = torch.float16,
    backend: Optional[str] = None,
    **model_kwargs,
) -> list[DimensionResult]:
    effective_epsilon_range = DEFAULT_EPSILON_RANGE if epsilon_range is None else epsilon_range

    curves = curve_from_texts(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        context_length=context_length,
        dim_reduction=dim_reduction,
        stride=stride,
        epsilon_range=effective_epsilon_range,
        num_epsilon=num_epsilon,
        block_size=block_size,
        show_progress=show_progress,
        precision=precision,
        backend=backend,
        **model_kwargs,
    )
    return [
        estimate_dimension_from_curve(
            curve,
            correlation_integral_range=correlation_integral_range,
            epsilon_range=epsilon_range,
        )
        for curve in curves
    ]
