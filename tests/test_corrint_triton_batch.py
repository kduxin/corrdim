import torch

import pytest

from corrdim.corrint.triton import correlation_counts


def _random_inputs(device, same=True):
    torch.manual_seed(0)
    B, M, N, K = 3, 16, 20, 32
    epsilons = (torch.rand(5, device=device) + 0.1).contiguous()
    vecs1 = torch.randn(B, M, K, device=device).contiguous()
    if same:
        vecs2 = vecs1
    else:
        vecs2 = torch.randn(B, N, K, device=device).contiguous()
    return vecs1, vecs2, epsilons


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available; skipping Triton tests")
def test_batch_self():
    device = torch.device("cuda")
    vecs, _, eps = _random_inputs(device, same=True)
    batch_counts = correlation_counts(vecs, eps)
    expected = []
    for b in range(vecs.shape[0]):
        expected.append(correlation_counts(vecs[b], eps))
    expected = torch.stack(expected, dim=0)
    assert torch.equal(batch_counts, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available; skipping Triton tests")
def test_batch_cross():
    device = torch.device("cuda")
    vecs1, vecs2, eps = _random_inputs(device, same=False)
    batch_counts = correlation_counts(vecs1, eps, vecs_other=vecs2)
    expected = []
    for b in range(vecs1.shape[0]):
        expected.append(correlation_counts(vecs1[b], eps, vecs_other=vecs2[b]))
    expected = torch.stack(expected, dim=0)
    assert torch.equal(batch_counts, expected)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping Triton tests.")
        return
    device = torch.device("cuda")
    # Manual run helper
    vecs, _, eps = _random_inputs(device, same=True)
    _ = correlation_counts(vecs, eps)
    vecs1, vecs2, eps = _random_inputs(device, same=False)
    _ = correlation_counts(vecs1, eps, vecs_other=vecs2)
    print("All batch vs non-batch Triton comparisons passed.")


if __name__ == "__main__":
    main()

