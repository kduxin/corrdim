"""
CorrDim: Correlation Dimension for Language Models

A library for computing correlation dimension of autoregressive large language models,
based on the research paper "Correlation Dimension of Auto-regressive Large Language Models" (NeurIPS 2025).
"""

__author__ = "duxin"
__email__ = "duxin@tongji.edu.cn"

from .corrint import (
    available_corrint_backends,
    correlation_counts,
    correlation_integral,
    progressive_correlation_counts,
    progressive_correlation_integral,
    set_corrint_backend,
)
from .dimension import (
    auto_linear_region_bounds,
    estimate_dimension_from_curve,
    estimate_dimension_from_curves,
)
from .high_level import measure_text, measure_texts
from .low_level import (
    clear_model_cache,
    curve_from_text,
    curve_from_texts,
    curve_from_vectors,
    curve_from_vectors_batch,
    progressive_curve_from_text,
    progressive_curve_from_texts,
    progressive_curve_from_vectors,
    progressive_curve_from_vectors_batch,
)
from .models import LanguageModelWrapper
from .types import CurveResult, DimensionResult, ProgressiveCurveResult
from .utils import clamp, reduce_dimension

__all__ = [
    "clamp",
    "available_corrint_backends",
    "correlation_counts",
    "correlation_integral",
    "progressive_correlation_counts",
    "progressive_correlation_integral",
    "curve_from_vectors",
    "curve_from_vectors_batch",
    "curve_from_text",
    "curve_from_texts",
    "progressive_curve_from_vectors",
    "progressive_curve_from_vectors_batch",
    "progressive_curve_from_text",
    "progressive_curve_from_texts",
    "clear_model_cache",
    "estimate_dimension_from_curve",
    "estimate_dimension_from_curves",
    "measure_text",
    "measure_texts",
    "CurveResult",
    "ProgressiveCurveResult",
    "DimensionResult",
    "auto_linear_region_bounds",
    "LanguageModelWrapper",
    "set_corrint_backend",
    "reduce_dimension",
]
