import torch

import pytest

from corrdim.corrint.triton import correlation_counts, progressive_correlation_counts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available; skipping Triton tests")
def test_triton_progressive_counts_matches_loop_definition():
    torch.manual_seed(0)
    device = torch.device("cuda")

    b, m, k = 2, 200, 16  # progressive tests use at least 200 sequence positions
    vecs = torch.randn(b, m, k, device=device)
    eps = torch.tensor([0.0, 0.5, 1.0, 2.0], device=device)

    got = progressive_correlation_counts(vecs, eps)
    assert got.shape == (b, m, eps.numel())
    assert torch.equal(got[:, 0, :], torch.zeros((b, eps.numel()), device=device, dtype=torch.int64))

    # Definition (matches the old implementation):
    # inc[i] = 2 * correlation_counts(vecs[:i], vecs_other=vecs[i:i+1])
    # counts = cumsum(inc, dim=1)
    inc = torch.zeros_like(got)
    for i in range(1, m):
        inc[:, i, :] = correlation_counts(vecs[:, :i, :], eps, vecs_other=vecs[:, i : i + 1, :]) * 2
    ref = inc.cumsum(dim=1)

    assert torch.equal(got.cpu(), ref.cpu())

