from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CurveResult:
    sequence_length: int
    epsilons: np.ndarray
    corrints: np.ndarray


@dataclass
class ProgressiveCurveResult:
    sequence_length: int
    epsilons: np.ndarray
    corrints_progressive: np.ndarray


@dataclass
class DimensionResult:
    sequence_length: int
    epsilons: np.ndarray
    corrints: np.ndarray
    corrdim: float
    fit_r2: float
    epsilons_linear_region: np.ndarray
    corrints_linear_region: np.ndarray
    linear_region_bounds: Tuple[Optional[float], Optional[float]] = (None, None)
