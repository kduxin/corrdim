import numpy as np
import pytest
import torch

from corrdim.utils import clamp, group_add, group_mean, reduce_dimension


def test_clamp_strict_bounds_numpy():
    # Strictly greater than low and strictly less than high.
    values = np.array([10, 20, 30, 40], dtype=np.float64)
    reference = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)

    kept_values, kept_ref = clamp(values, reference, low=0.25, high=0.75)

    # reference==low and reference==high are excluded.
    assert kept_values.tolist() == [30]
    assert kept_ref.tolist() == [0.5]


def test_group_add_num_groups_validation():
    vecs = torch.randn(5, 7)
    with pytest.raises(ValueError, match="num_groups must be a positive integer"):
        _ = group_add(vecs, num_groups=0)


def test_group_add_matches_manual_groups():
    # Each vocab index v maps to group (v % num_groups).
    vecs = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6)
    num_groups = 4

    got = group_add(vecs, num_groups=num_groups)
    expected = torch.zeros(2, num_groups, dtype=vecs.dtype)
    for v in range(vecs.shape[-1]):
        g = v % num_groups
        expected[:, g] += vecs[:, v]

    assert torch.equal(got, expected)


def test_group_mean_matches_manual_groups():
    # group_mean should equal group_add / count_per_group (with clamp_min(1)).
    torch.manual_seed(0)
    vecs = torch.randn(3, 9)
    num_groups = 4

    got = group_mean(vecs, num_groups=num_groups)

    counts = torch.zeros(num_groups, dtype=torch.int64)
    for v in range(vecs.shape[-1]):
        counts[v % num_groups] += 1
    counts_clamped = counts.clamp_min(1).to(dtype=vecs.dtype)

    expected = group_add(vecs, num_groups=num_groups) / counts_clamped.view(1, -1)
    assert torch.allclose(got, expected, atol=0, rtol=0)


def test_reduce_dimension_dispatch_and_invalid_method():
    torch.manual_seed(0)
    vecs = torch.randn(2, 10)
    out_add = reduce_dimension(vecs, num_groups=3, method="group_add")
    out_mean = reduce_dimension(vecs, num_groups=3, method="group_mean")
    assert out_add.shape == (2, 3)
    assert out_mean.shape == (2, 3)

    with pytest.raises(ValueError, match="Invalid method"):
        _ = reduce_dimension(vecs, num_groups=3, method="nope")


def test_group_add_accepts_numpy_input():
    vecs_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = group_add(vecs_np, num_groups=2)
    assert isinstance(got, torch.Tensor)
    assert got.shape == (3, 2)

