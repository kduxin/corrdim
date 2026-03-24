from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import sklearn.linear_model

from .types import CurveResult, DimensionResult
from .utils import clamp


def auto_linear_region_bounds(sequence_length: int, epsilons: np.ndarray, corrints: np.ndarray) -> Tuple[float, float]:
    n = sequence_length
    for low, high in [
        (20.0 / n / (n - 1), 1.0 / n),
        (20.0 / n / (n - 1), 1.0 / n * 10),
        (20.0 / n / (n - 1), 1.0 / n * 100),
        (0.0, 1.0),
    ]:
        eps_filtered, _corr_filtered = clamp(epsilons, corrints, low=low, high=high)
        if len(eps_filtered) >= 2:
            return low, high
    raise ValueError("Not enough points in the selected correlation-integral range.")


def estimate_dimension_from_curve(
    curve: CurveResult,
    correlation_integral_range: Optional[Union[str, Tuple[float, float]]] = None,
    epsilon_range: Optional[Tuple[float, float]] = None,
) -> DimensionResult:
    epsilons = curve.epsilons.copy()
    corrints = curve.corrints.copy()

    if correlation_integral_range is not None:
        if correlation_integral_range == "auto":
            try:
                low, high = auto_linear_region_bounds(curve.sequence_length, epsilons, corrints)
            except ValueError:
                low, high = 0.0, 1.0
        else:
            low, high = correlation_integral_range
        eps_linear, corr_linear = clamp(epsilons, corrints, low=low, high=high)
    else:
        low, high = None, None
        eps_linear, corr_linear = epsilons, corrints

    if epsilon_range is not None:
        corr_linear, eps_linear = clamp(corr_linear, eps_linear, low=epsilon_range[0], high=epsilon_range[1])

    valid = np.isfinite(corr_linear) & (corr_linear > 0)
    if valid.sum() < 2:
        # Fallback to the full curve if selected range is too narrow (common on short/noisy sequences).
        eps_linear = epsilons
        corr_linear = corrints
        valid = np.isfinite(corr_linear) & (corr_linear > 0)
        if valid.sum() < 2:
            raise ValueError("Not enough points with positive finite correlation integral after range selection.")

    eps_fit = eps_linear[valid]
    corr_fit = corr_linear[valid]
    log_eps = np.log10(eps_fit).reshape(-1, 1)
    log_corr = np.log10(corr_fit)

    fit = sklearn.linear_model.LinearRegression().fit(log_eps, log_corr)
    fit_r2 = fit.score(log_eps, log_corr)

    return DimensionResult(
        sequence_length=curve.sequence_length,
        epsilons=curve.epsilons,
        corrints=curve.corrints,
        corrdim=float(fit.coef_[0]),
        fit_r2=float(fit_r2),
        epsilons_linear_region=eps_fit,
        corrints_linear_region=corr_fit,
        linear_region_bounds=(low, high),
    )


def estimate_dimension_from_curves(
    curves: list[CurveResult],
    correlation_integral_range: Optional[Union[str, Tuple[float, float]]] = "auto",
    epsilon_range: Optional[Tuple[float, float]] = None,
) -> list[DimensionResult]:
    return [
        estimate_dimension_from_curve(
            curve,
            correlation_integral_range=correlation_integral_range,
            epsilon_range=epsilon_range,
        )
        for curve in curves
    ]
