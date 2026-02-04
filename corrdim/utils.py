"""
Utility functions for the corrdim library.
"""

from typing import Tuple, Union

import numpy as np
import torch

Tensor = Union[torch.Tensor, np.ndarray]


def clamp(values: Tensor, reference: Tensor, low: float, high: float) -> Tuple[Tensor, Tensor]:
    """Filter values by a reference range, returning filtered values and reference."""
    in_range = (reference > low) & (reference < high)
    return values[in_range], reference[in_range]
