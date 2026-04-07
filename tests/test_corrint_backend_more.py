import pytest
import torch

import corrdim
from corrdim.corrint import available_corrint_backends


def test_available_corrint_backends_shape_and_keys():
    out = available_corrint_backends()
    assert set(out.keys()) == {"cuda", "triton", "pytorch", "pytorch_fast"}
    assert all(isinstance(v, bool) for v in out.values())


def test_set_corrint_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown backend"):
        corrdim.set_corrint_backend("not-a-backend")


def test_correlation_integral_return_counts_matches_correlation_counts():
    corrdim.set_corrint_backend("pytorch")
    torch.manual_seed(0)
    vecs = torch.randn(10, 4)
    eps = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    counts = corrdim.correlation_counts(vecs, eps, backend="pytorch")
    got_counts = corrdim.correlation_integral(vecs, eps, return_counts=True, backend="pytorch")
    assert torch.equal(got_counts, counts)


def test_progressive_integral_return_counts_matches_progressive_counts():
    corrdim.set_corrint_backend("pytorch")
    torch.manual_seed(1)
    vecs = torch.randn(200, 3)  # progressive tests use at least 100 sequence positions
    eps = torch.tensor([0.25, 1.5], dtype=torch.float32)

    counts = corrdim.progressive_correlation_counts(vecs, eps, backend="pytorch")
    got_counts = corrdim.progressive_correlation_integral(vecs, eps, return_counts=True, backend="pytorch")
    assert torch.equal(got_counts, counts)

