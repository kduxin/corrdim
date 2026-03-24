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

def reduce_dimension(vectors: Tensor, num_groups: int = 8192, method: str = "group_add") -> Tensor:
    if method == "group_add":
        return group_add(vectors, num_groups)
    elif method == "group_mean":
        return group_mean(vectors, num_groups)
    else:
        raise ValueError(f"Invalid method: {method}")

def group_add(vectors: Tensor, num_groups: int) -> Tensor:
    if not isinstance(vectors, torch.Tensor):
        vectors = torch.as_tensor(vectors)
    if num_groups <= 0:
        raise ValueError("num_groups must be a positive integer")

    vocab_size = vectors.shape[-1]
    group_index = torch.arange(vocab_size, device=vectors.device) % num_groups
    scatter_index = group_index.view(*([1] * (vectors.dim() - 1)), vocab_size).expand_as(vectors)

    reduced = torch.zeros(*vectors.shape[:-1], num_groups, dtype=vectors.dtype, device=vectors.device)
    reduced.scatter_add_(-1, scatter_index, vectors)
    return reduced

def group_mean(vectors: Tensor, num_groups: int) -> Tensor:
    if not isinstance(vectors, torch.Tensor):
        vectors = torch.as_tensor(vectors)
    vocab_size = vectors.shape[-1]
    reduced = group_add(vectors, num_groups)
    counts = torch.bincount(
        torch.arange(vocab_size, device=reduced.device) % num_groups,
        minlength=num_groups,
    ).to(dtype=reduced.dtype)
    counts = counts.clamp_min(1)
    return reduced / counts.view(*([1] * (reduced.dim() - 1)), num_groups)