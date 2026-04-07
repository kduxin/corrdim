import torch

import pytest

from corrdim.corrint import (
    correlation_counts,
    correlation_integral,
    progressive_correlation_counts,
    progressive_correlation_integral,
)


def _brute_counts_self(vecs: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    # Ordered pairs (i, j), i != j.
    d = torch.cdist(vecs.to(torch.float32), vecs.to(torch.float32), p=2)
    m = d.shape[0]
    d.fill_diagonal_(float("inf"))
    # (T,)
    return torch.stack([(d <= e).sum().to(torch.int64) for e in eps], dim=0)


def _brute_counts_cross(vecs1: torch.Tensor, vecs2: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(vecs1.to(torch.float32), vecs2.to(torch.float32), p=2)
    return torch.stack([(d <= e).sum().to(torch.int64) for e in eps], dim=0)


@pytest.mark.parametrize("backend", ["pytorch", "pytorch_fast"])
def test_pytorch_counts_self_matches_bruteforce(backend: str):
    torch.manual_seed(0)
    vecs = torch.randn(12, 7, device="cpu")
    eps = torch.tensor([0.0, 0.5, 1.0, 2.0], device="cpu")

    got = correlation_counts(vecs, eps, backend=backend)
    ref = _brute_counts_self(vecs, eps)
    assert torch.equal(got.cpu(), ref.cpu())


@pytest.mark.parametrize("backend", ["pytorch", "pytorch_fast"])
def test_pytorch_counts_cross_matches_bruteforce(backend: str):
    torch.manual_seed(1)
    vecs1 = torch.randn(10, 5, device="cpu")
    vecs2 = torch.randn(8, 5, device="cpu")
    eps = torch.tensor([0.0, 0.25, 1.25], device="cpu")

    got = correlation_counts(vecs1, eps, vecs_other=vecs2, backend=backend)
    ref = _brute_counts_cross(vecs1, vecs2, eps)
    assert torch.equal(got.cpu(), ref.cpu())


@pytest.mark.parametrize("backend", ["pytorch", "pytorch_fast"])
def test_pytorch_integral_self_normalization(backend: str):
    torch.manual_seed(2)
    vecs = torch.randn(9, 4, device="cpu")
    eps = torch.tensor([0.75], device="cpu")

    counts = correlation_counts(vecs, eps, backend=backend).to(torch.float32)
    got = correlation_integral(vecs, eps, backend=backend)
    ref = counts / float(vecs.shape[0] * (vecs.shape[0] - 1))
    assert torch.allclose(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["pytorch", "pytorch_fast"])
def test_pytorch_integral_cross_normalization(backend: str):
    torch.manual_seed(3)
    vecs1 = torch.randn(7, 4, device="cpu")
    vecs2 = torch.randn(6, 4, device="cpu")
    eps = torch.tensor([1.0, 2.0], device="cpu")

    counts = correlation_counts(vecs1, eps, vecs_other=vecs2, backend=backend).to(torch.float32)
    got = correlation_integral(vecs1, eps, vecs_other=vecs2, backend=backend)
    ref = counts / float(vecs1.shape[0] * vecs2.shape[0])
    assert torch.allclose(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["pytorch", "pytorch_fast"])
def test_pytorch_batched_matches_per_batch(backend: str):
    torch.manual_seed(4)
    b, m, k = 3, 10, 6
    vecs = torch.randn(b, m, k, device="cpu")
    eps = torch.tensor([0.5, 1.0], device="cpu")

    got = correlation_counts(vecs, eps, backend=backend)
    ref = torch.stack([correlation_counts(vecs[i], eps, backend=backend) for i in range(b)], dim=0)
    assert torch.equal(got.cpu(), ref.cpu())


@pytest.mark.parametrize("backend", ["pytorch", "pytorch_fast"])
def test_pytorch_progressive_counts_matches_definition(backend: str):
    torch.manual_seed(5)
    m = 200  # progressive tests use at least 200 sequence positions
    vecs = torch.randn(2, m, 4, device="cpu")  # (B, M, K)
    eps = torch.tensor([0.0, 1.0], device="cpu")

    got = progressive_correlation_counts(vecs, eps, backend=backend)
    assert got.shape == (2, m, 2)
    assert torch.equal(got[:, 0, :], torch.zeros((2, 2), dtype=torch.int64))

    # Definition (matches Triton progressive implementation):
    # inc[i] = 2 * correlation_counts(vecs[:i], vecs_other=vecs[i:i+1])
    # counts = cumsum(inc, dim=1)
    inc = torch.zeros_like(got)
    for i in range(1, vecs.shape[1]):
        inc[:, i, :] = correlation_counts(
            vecs[:, :i, :],
            eps,
            vecs_other=vecs[:, i : i + 1, :],
            backend=backend,
        ) * 2
    ref = inc.cumsum(dim=1)
    assert torch.equal(got.cpu(), ref.cpu())


@pytest.mark.parametrize("backend", ["pytorch", "pytorch_fast"])
def test_pytorch_progressive_integral_shape_and_finite(backend: str):
    torch.manual_seed(6)
    m = 200
    vecs = torch.randn(m, 5, device="cpu")
    eps = torch.tensor([0.5, 1.0, 2.0], device="cpu")

    got = progressive_correlation_integral(vecs, eps, backend=backend)
    assert got.shape == (m, 3)
    assert torch.isfinite(got).all()

