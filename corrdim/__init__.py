"""
CorrDim: Correlation Dimension for Language Models

A library for computing correlation dimension of autoregressive large language models,
based on the research paper "Correlation Dimension of Auto-regressive Large Language Models" (NeurIPS 2025).
"""

__author__ = "duxin"
__email__ = "duxin.ac@gmail.com"

from .calculator import CorrelationDimensionCalculator, CorrelationIntegralResult, CorrelationDimensionResult
from .models import LanguageModelWrapper
from .utils import load_model, get_log_probabilities
from .corrint import set_corrint_backend, correlation_integral

__all__ = [
    "set_corrint_backend",
    "correlation_integral",
    "CorrelationDimensionCalculator",
    "LanguageModelWrapper", 
    "load_model",
    "get_log_probabilities",
    "CorrelationIntegralResult",
    "CorrelationDimensionResult",
]
