"""
Utility functions for the corrdim library.
"""

import torch
import numpy as np
from typing import Union

Tensor = Union[torch.Tensor, np.ndarray]

def clamp(values: Tensor, reference: Tensor, low: float, high: float) -> Tensor:
    in_range = (reference > low) & (reference < high)
    return values[in_range], reference[in_range]
