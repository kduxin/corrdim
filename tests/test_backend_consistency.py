import pytest
import os
import torch

import corrdim
from corrdim.corrint import available_corrint_backends


def _make_inputs(device: torch.device, *, m: int = 32, n: int = 32, k: int = 16):
    torch.manual_seed(0)
    vecs = torch.randn(m, k, device=device, dtype=torch.float32).contiguous()
    vecs_other = torch.randn(n, k, device=device, dtype=torch.float32).contiguous()
    # Choose epsilons around typical Euclidean distances (~sqrt(2k)).
    # Use a huge epsilon to validate the "all pairs" saturation behavior.
    eps = torch.tensor([1e-3, 3.0, 6.0, 1e6], device=device, dtype=torch.float32).contiguous()
    return vecs, vecs_other, eps


def _assert_counts_closeish(got: torch.Tensor, ref: torch.Tensor, *, max_abs_diff: int = 1):
    # Counts are integers; small numerical boundary shifts can change counts by +/- 1.
    diff = (got.to(torch.int64) - ref.to(torch.int64)).abs().max().item()
    assert diff <= max_abs_diff, f"max abs diff in counts too large: {diff} > {max_abs_diff}"


def _assert_integral_closeish(got: torch.Tensor, ref: torch.Tensor, *, atol: float):
    torch.testing.assert_close(got, ref, rtol=0.0, atol=atol)


def test_pytorch_vs_pytorch_fast_self_and_cross_consistent_cpu():
    # These should match exactly, and are already validated against brute force elsewhere.
    device = torch.device("cpu")
    vecs, vecs_other, eps = _make_inputs(device, m=28, n=24, k=16)

    ref_counts = corrdim.correlation_counts(vecs, eps, backend="pytorch")
    fast_counts = corrdim.correlation_counts(vecs, eps, backend="pytorch_fast")
    assert torch.equal(ref_counts, fast_counts)

    ref_integral = corrdim.correlation_integral(vecs, eps, backend="pytorch")
    fast_integral = corrdim.correlation_integral(vecs, eps, backend="pytorch_fast")
    _assert_integral_closeish(fast_integral, ref_integral, atol=0.0)

    ref_counts_cross = corrdim.correlation_counts(vecs, eps, vecs_other=vecs_other, backend="pytorch")
    fast_counts_cross = corrdim.correlation_counts(vecs, eps, vecs_other=vecs_other, backend="pytorch_fast")
    assert torch.equal(ref_counts_cross, fast_counts_cross)

    ref_integral_cross = corrdim.correlation_integral(vecs, eps, vecs_other=vecs_other, backend="pytorch")
    fast_integral_cross = corrdim.correlation_integral(vecs, eps, vecs_other=vecs_other, backend="pytorch_fast")
    _assert_integral_closeish(fast_integral_cross, ref_integral_cross, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available; skipping CUDA backend consistency checks")
def test_cuda_backend_matches_pytorch_self_cross_and_progressive():
    availability = available_corrint_backends()
    if not availability.get("cuda", False):
        pytest.skip("CUDA backend not available")

    device = torch.device("cuda")
    # Progressive checks need at least 100 sequence positions.
    vecs, vecs_other, eps = _make_inputs(device, m=200, n=200, k=16)

    # In some dev environments, the CUDA extension build may fail early (e.g. CUDA_HOME not set),
    # causing the CUDA backend to fall back to pytorch and emit a RuntimeWarning.
    # For backend consistency checks, that's expected, and we don't want the test output polluted.
    cuda_home_missing = not os.environ.get("CUDA_HOME")
    if cuda_home_missing:
        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r"^corrdim\.corrint\.cuda backend fell back to pytorch backend\..*",
            category=RuntimeWarning,
        )

    # Self
    ref_counts = corrdim.correlation_counts(vecs, eps, backend="pytorch")
    got_counts = corrdim.correlation_counts(vecs, eps, backend="cuda")
    _assert_counts_closeish(got_counts, ref_counts, max_abs_diff=1)

    ref_integral = corrdim.correlation_integral(vecs, eps, backend="pytorch")
    got_integral = corrdim.correlation_integral(vecs, eps, backend="cuda")
    # One-count difference means ~1/(m*(m-1)). For m=100 => ~1e-4.
    _assert_integral_closeish(got_integral, ref_integral, atol=2e-3)

    # Cross
    ref_counts_cross = corrdim.correlation_counts(vecs, eps, vecs_other=vecs_other, backend="pytorch")
    got_counts_cross = corrdim.correlation_counts(vecs, eps, vecs_other=vecs_other, backend="cuda")
    _assert_counts_closeish(got_counts_cross, ref_counts_cross, max_abs_diff=1)

    ref_integral_cross = corrdim.correlation_integral(vecs, eps, vecs_other=vecs_other, backend="pytorch")
    got_integral_cross = corrdim.correlation_integral(vecs, eps, vecs_other=vecs_other, backend="cuda")
    # One-count difference means ~1/(m*n). For 100x100 => ~1e-4.
    _assert_integral_closeish(got_integral_cross, ref_integral_cross, atol=2e-3)

    # Progressive (exclude prefix-length index 0 due to different denom choice across implementations)
    ref_prog_counts = corrdim.progressive_correlation_counts(vecs, eps, backend="pytorch")
    got_prog_counts = corrdim.progressive_correlation_counts(vecs, eps, backend="cuda")
    _assert_counts_closeish(got_prog_counts, ref_prog_counts, max_abs_diff=1)

    ref_prog_integral = corrdim.progressive_correlation_integral(vecs, eps, backend="pytorch")
    got_prog_integral = corrdim.progressive_correlation_integral(vecs, eps, backend="cuda")
    _assert_integral_closeish(got_prog_integral[1:, :], ref_prog_integral[1:, :], atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available; skipping Triton backend consistency checks")
def test_triton_backend_matches_pytorch_self_cross_and_progressive():
    availability = available_corrint_backends()
    if not availability.get("triton", False):
        pytest.skip("Triton backend not available")

    device = torch.device("cuda")
    # Progressive checks need at least 100 sequence positions.
    vecs, vecs_other, eps = _make_inputs(device, m=100, n=100, k=16)

    # Self
    ref_counts = corrdim.correlation_counts(vecs, eps, backend="pytorch")
    got_counts = corrdim.correlation_counts(vecs, eps, backend="triton")
    _assert_counts_closeish(got_counts, ref_counts, max_abs_diff=1)

    ref_integral = corrdim.correlation_integral(vecs, eps, backend="pytorch")
    got_integral = corrdim.correlation_integral(vecs, eps, backend="triton")
    _assert_integral_closeish(got_integral, ref_integral, atol=2e-3)

    # Cross
    ref_counts_cross = corrdim.correlation_counts(vecs, eps, vecs_other=vecs_other, backend="pytorch")
    got_counts_cross = corrdim.correlation_counts(vecs, eps, vecs_other=vecs_other, backend="triton")
    _assert_counts_closeish(got_counts_cross, ref_counts_cross, max_abs_diff=1)

    ref_integral_cross = corrdim.correlation_integral(vecs, eps, vecs_other=vecs_other, backend="pytorch")
    got_integral_cross = corrdim.correlation_integral(vecs, eps, vecs_other=vecs_other, backend="triton")
    _assert_integral_closeish(got_integral_cross, ref_integral_cross, atol=2e-3)

    # Progressive (exclude prefix-length index 0 due to different denom choice across implementations)
    ref_prog_counts = corrdim.progressive_correlation_counts(vecs, eps, backend="pytorch")
    got_prog_counts = corrdim.progressive_correlation_counts(vecs, eps, backend="triton")
    _assert_counts_closeish(got_prog_counts, ref_prog_counts, max_abs_diff=1)

    ref_prog_integral = corrdim.progressive_correlation_integral(vecs, eps, backend="pytorch")
    got_prog_integral = corrdim.progressive_correlation_integral(vecs, eps, backend="triton")
    _assert_integral_closeish(got_prog_integral[1:, :], ref_prog_integral[1:, :], atol=2e-3)

