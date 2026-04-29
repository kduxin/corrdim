import torch

import pytest

try:
    from corrdim.corrint.triton import correlation_counts
except Exception:
    correlation_counts = None

pytestmark = pytest.mark.skipif(correlation_counts is None, reason="Triton not available")


def _random_noncontig_slices(device):
    torch.manual_seed(0)
    # In some Triton versions/architectures, tl.dot requires M/N/K >= 16.
    # Ensure dimensions stay >= 16 after two rounds of slicing (::2, then half).
    B, M, N, K = 3, 64, 64, 32
    epsilons = (torch.rand(5, device=device) + 0.1).contiguous()
    vecs1 = torch.randn(B, M, K, device=device).contiguous()
    vecs2 = torch.randn(B, N, K, device=device).contiguous()

    # Create non-contiguous views via strided slicing.
    v1_view = vecs1[:, ::2, :]              # (B, M/2, K), non-contiguous
    v2_view = vecs2[:, 1::2, :]             # (B, N/2, K), non-contiguous

    # Random index slicing (typically also non-contiguous)
    idx1 = torch.randperm(v1_view.shape[1], device=device)[: v1_view.shape[1] // 2]
    idx2 = torch.randperm(v2_view.shape[1], device=device)[: v2_view.shape[1] // 2]
    v1_rand = v1_view.index_select(1, idx1)
    v2_rand = v2_view.index_select(1, idx2)

    return (v1_view, v2_view, epsilons), (v1_rand, v2_rand, epsilons)


def _assert_batch_matches_nonbatch(vecs1, vecs2, eps):
    batch_counts = correlation_counts(vecs1, eps, vecs_other=vecs2)
    expected = []
    for b in range(vecs1.shape[0]):
        expected.append(correlation_counts(vecs1[b].contiguous(), eps, vecs_other=vecs2[b].contiguous()))
    expected = torch.stack(expected, dim=0)
    assert torch.equal(batch_counts, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available; skipping Triton tests")
def test_batch_cross_noncontig():
    device = torch.device("cuda")
    (v1_view, v2_view, eps), _ = _random_noncontig_slices(device)
    _assert_batch_matches_nonbatch(v1_view, v2_view, eps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available; skipping Triton tests")
def test_batch_cross_random_slice():
    device = torch.device("cuda")
    _, (v1_rand, v2_rand, eps) = _random_noncontig_slices(device)
    _assert_batch_matches_nonbatch(v1_rand, v2_rand, eps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available; skipping Triton tests")
def test_batch_self_noncontig():
    device = torch.device("cuda")
    torch.manual_seed(1)
    B, M, K = 3, 32, 32
    epsilons = (torch.rand(5, device=device) + 0.1).contiguous()
    vecs = torch.randn(B, M, K, device=device).contiguous()
    view = vecs[:, 1::2, :]  # non-contiguous

    batch_counts = correlation_counts(view, epsilons)
    expected = []
    for b in range(view.shape[0]):
        expected.append(correlation_counts(view[b].contiguous(), epsilons))
    expected = torch.stack(expected, dim=0)
    assert torch.equal(batch_counts, expected)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping Triton tests.")
        return
    device = torch.device("cuda")
    # Manual run helper
    (v1_view, v2_view, eps), _ = _random_noncontig_slices(device)
    _ = correlation_counts(v1_view, eps, vecs_other=v2_view)
    torch.manual_seed(1)
    B, M, K = 3, 32, 32
    epsilons = (torch.rand(5, device=device) + 0.1).contiguous()
    vecs = torch.randn(B, M, K, device=device).contiguous()
    view = vecs[:, 1::2, :]
    _ = correlation_counts(view, epsilons)
    print("All batch slice vs non-batch Triton comparisons passed.")


if __name__ == "__main__":
    main()
