import pytest
import torch

from corrdim.corrint import cuda as cuda_impl
from corrdim.corrint import pytorch as pytorch_impl


def test_cuda_backend_fallback_counts_warns_and_matches_pytorch(monkeypatch):
    monkeypatch.setattr(cuda_impl, "_FALLBACK_WARNED", False)

    vecs = torch.randn(9, 4, device="cpu")
    vecs_other = torch.randn(7, 4, device="cpu")
    eps = torch.tensor([0.0, 0.5, 1.0], device="cpu")

    with pytest.warns(RuntimeWarning, match="fell back to pytorch backend"):
        got = cuda_impl.correlation_counts(vecs, eps, vecs_other=vecs_other, block_size=512, show_progress=False)

    ref = pytorch_impl.correlation_counts(
        vecs,
        eps,
        vecs_other=vecs_other,
        block_size=512,
        show_progress=False,
        fast=True,
    )
    assert torch.equal(got.cpu(), ref.cpu())


def test_cuda_backend_fallback_progressive_integral_warns_and_matches_pytorch(monkeypatch):
    monkeypatch.setattr(cuda_impl, "_FALLBACK_WARNED", False)

    vecs = torch.randn(10, 5, device="cpu")
    eps = torch.tensor([0.25, 1.25], device="cpu")

    with pytest.warns(RuntimeWarning, match="fell back to pytorch backend"):
        got = cuda_impl.progressive_correlation_integral(vecs, eps, block_size=512, show_progress=False)

    ref = pytorch_impl.progressive_correlation_integral(vecs, eps, block_size=512, show_progress=False, fast=True)
    assert torch.allclose(got.cpu(), ref.cpu(), atol=0, rtol=0)

