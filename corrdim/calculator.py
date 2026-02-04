"""
Core correlation dimension calculation module.

This module implements the main correlation dimension algorithm as described in the paper.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import tqdm.auto as tqdm
import torch
import sklearn.linear_model
from dataclasses import dataclass

from .utils import clamp
from .models import LanguageModelWrapper
from .corrint import correlation_integral


@dataclass
class CorrelationIntegralResult:
    sequence_length: int
    epsilons: np.ndarray
    corrints: np.ndarray


@dataclass
class CorrelationDimensionResult:
    sequence_length: int
    epsilons: np.ndarray
    corrints: np.ndarray
    corrdim: float
    fit_r2: float
    epsilons_linear_region: np.ndarray
    corrints_linear_region: np.ndarray
    linear_region_bounds: Tuple[float, float]


class CorrelationDimensionCalculator:
    """
    Main class for computing correlation dimension from language model log-probabilities.

    The correlation dimension quantifies self-similarity in sequences by analyzing
    their recurrence structure. It measures the scaling behavior of the correlation
    integral S(ε) as a function of distance threshold ε.
    """

    def __init__(
        self,
        model: Union[str, LanguageModelWrapper],
        tokenizer: Optional[object] = None,
        context_length: int = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the correlation dimension calculator.

        Args:
            model: Language model (name string or LanguageModelWrapper instance)
            tokenizer: Tokenizer for the model (if model is a string)
            context_length: Maximum context length for the model
            device: Device to run computations on ('cpu', 'cuda', etc.)
        """
        if isinstance(model, str):
            from .models import create_model_wrapper

            self.model_wrapper = create_model_wrapper(model, tokenizer=tokenizer, device=device, **kwargs)
        else:
            self.model_wrapper = model

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def get_log_probability(
        self,
        text: str,
        dim_reduction: int = None,
        context_length: int = None,
        stride: Union[int, str] = "auto",
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Extract log-probability vectors for each token position in the text.

        Args:
            text: Input text to analyze
            stride: Stride for sliding window. Can be:
                - int: Fixed stride value
                - "auto": Automatically set stride to max(1, context_length // 10)

        Returns:
            Array of log-probability vectors of shape (seq_len, vocab_size)
        """
        return self.model_wrapper.get_log_probabilities(
            text,
            dim_reduction=dim_reduction,
            context_length=context_length,
            stride=stride,
            show_progress=show_progress,
        )

    def compute_correlation_integral_curve(
        self,
        text: str,
        context_length: int = None,
        dim_reduction: int = None,
        stride: Union[int, str] = "auto",
        epsilon_range: Optional[Tuple[float, float]] = (10**-20.0, 10**20.0),
        num_epsilon: int = 1000,
        block_size: int = 512,
        show_progress: bool = False,
        precision: str = torch.float32,
    ) -> CorrelationIntegralResult:
        """
        Compute the full correlation integral curve S(ε) vs ε.

        Args:
            text: Input text to analyze
            epsilon_log10_range: Range of epsilon values in log10 scale (min, max)
            num_epsilon: Number of epsilon values to test
            stride: Stride for sliding window. Can be:
                - int: Fixed stride value
                - "auto": Automatically set stride to max(1, context_length // 10)
            show_progress: Whether to show progress bar
            precision: Precision for torch tensors

        Returns:
            Tuple of (epsilon_values, corrints)
        """
        # Get log-probability vectors
        log_probs = self.get_log_probability(
            text, context_length=context_length, dim_reduction=dim_reduction, stride=stride, show_progress=show_progress
        ).type(precision)
        return self.compute_correlation_integral_curve_from_vector_sequences(
            log_probs,
            epsilon_range=epsilon_range,
            num_epsilon=num_epsilon,
            block_size=block_size,
            show_progress=show_progress,
        )

    @staticmethod
    def compute_correlation_integral_curve_from_vector_sequences(
        vecs: torch.Tensor,
        epsilon_range: Optional[Tuple[float, float]] = (10**-20.0, 10**20.0),
        num_epsilon: int = 1024,
        block_size: int = 512,
        show_progress: bool = False,
    ) -> CorrelationIntegralResult:

        assert torch.isfinite(vecs).all(), "Found nan or inf in log-probability vectors. Please check the dtype of the model."

        assert vecs.shape[0] > 100, f"The sequence length is too short ({vecs.shape[0]} tokens). Please consider using a longer sequence with at least 100 tokens."

        N = vecs.shape[0]

        # Generate epsilon values (logarithmically spaced)
        epsilons = torch.logspace(
            np.log10(float(epsilon_range[0])),
            np.log10(float(epsilon_range[1])),
            num_epsilon,
            device=vecs.device,
        )

        corrints = correlation_integral(vecs, epsilons)
        epsilons, corrints = clamp(epsilons, corrints, low=corrints.min(), high=0.95)
        return CorrelationIntegralResult(N, epsilons.data.cpu().numpy(), corrints.data.cpu().numpy())

    def __call__(
        self,
        text: str,
        context_length: int = None,
        dim_reduction: int = None,
        correlation_integral_range: Optional[Union[str, Tuple[float, float]]] = "auto",
        epsilon_range: Optional[Tuple[float, float]] = (10**-20, 10**20),
        num_epsilon: int = 1000,
        stride: Union[int, str] = "auto",
        block_size: int = 512,
        show_progress: bool = False,
        precision: str = torch.float32,
    ) -> CorrelationDimensionResult:
        """
        Estimate the correlation dimension from the correlation integral curve.
        """
        corr_int_result = self.compute_correlation_integral_curve(
            text,
            context_length=context_length,
            dim_reduction=dim_reduction,
            epsilon_range=epsilon_range,
            num_epsilon=num_epsilon,
            stride=stride,
            block_size=block_size,
            show_progress=show_progress,
            precision=precision,
        )

        if correlation_integral_range == "auto":
            low, high = auto_linear_region_bounds(corr_int_result.sequence_length, corr_int_result.epsilons, corr_int_result.corrints)
        else:
            low, high = correlation_integral_range
        epsilons, corrints = clamp(corr_int_result.epsilons, corr_int_result.corrints, low=low, high=high)

        fit = sklearn.linear_model.LinearRegression().fit(np.log10(epsilons).reshape(-1, 1), np.log10(corrints))
        r2 = fit.score(np.log10(epsilons).reshape(-1, 1), np.log10(corrints))
        return CorrelationDimensionResult(
            sequence_length=corr_int_result.sequence_length,
            epsilons=corr_int_result.epsilons,
            corrints=corr_int_result.corrints,
            corrdim=fit.coef_[0],
            fit_r2=r2,
            epsilons_linear_region=epsilons,
            corrints_linear_region=corrints,
            linear_region_bounds=(low, high),
        )

    @staticmethod
    def compute_correlation_dimension_from_curve(
        epsilons: np.ndarray,
        corrints: np.ndarray,
        sequence_length: int,
        correlation_integral_range: Optional[Union[str, Tuple[float, float]]] = "auto",
    ) -> CorrelationDimensionResult:
        """Compute correlation dimension directly from a correlation integral curve."""

        if correlation_integral_range == "auto":
            low, high = auto_linear_region_bounds(sequence_length, epsilons, corrints)
        else:
            low, high = correlation_integral_range

        eps_filtered, corr_filtered = clamp(epsilons, corrints, low=low, high=high)

        log_eps = np.log10(eps_filtered).reshape(-1, 1)
        log_corr = np.log10(corr_filtered)

        fit = sklearn.linear_model.LinearRegression().fit(log_eps, log_corr)
        r2 = fit.score(log_eps, log_corr)

        return CorrelationDimensionResult(
            sequence_length=sequence_length,
            epsilons=eps_filtered,
            corrints=corr_filtered,
            corrdim=fit.coef_[0],
            fit_r2=r2,
            epsilons_linear_region=eps_filtered,
            corrints_linear_region=corr_filtered,
            linear_region_bounds=(low, high),
        )

    @property
    def tokenizer(self):
        return self.model_wrapper.tokenizer


def auto_linear_region_bounds(sequence_length: int, epsilons: np.ndarray, corrints: np.ndarray) -> Tuple[float, float]:
    N = sequence_length
    for low, high in [
        (20.0 / N / (N - 1), 1.0 / N),
        (20.0 / N / (N - 1), 1.0 / N * 10),
        (20.0 / N / (N - 1), 1.0 / N * 100),
        (0.0, 1.0),
    ]:
        eps_filtered, corr_filtered = clamp(epsilons, corrints, low=low, high=high)
        if len(eps_filtered) >= 2:
            break
    else:
        raise ValueError("Not enough points in the selected correlation-integral range.")
    return low, high
