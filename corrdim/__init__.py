"""
CorrDim: Correlation Dimension for Language Models

A library for computing correlation dimension of autoregressive large language models,
based on the research paper "Correlation Dimension of Auto-regressive Large Language Models" (NeurIPS 2025).
"""

__author__ = "duxin"
__email__ = "duxin.ac@gmail.com"

from .calculator import (
    CorrelationDimensionCalculator,
    CorrelationDimensionResult,
    CorrelationIntegralResult,
)
from .corrint import (
    available_corrint_backends,
    correlation_counts,
    correlation_integral,
    progressive_correlation_counts,
    progressive_correlation_integral,
    set_corrint_backend,
)
from .models import LanguageModelWrapper
from .utils import clamp

__all__ = [
    "clamp",
    "available_corrint_backends",
    "correlation_counts",
    "correlation_integral",
    "progressive_correlation_counts",
    "progressive_correlation_integral",
    "CorrelationDimensionCalculator",
    "CorrelationDimensionResult",
    "LanguageModelWrapper",
    "CorrelationIntegralResult",
    "set_corrint_backend",
]
