import pytest
import torch

import corrdim
from corrdim.corrint import available_corrint_backends

try:
    from corrdim.corrint import triton as triton_impl
except Exception:
    triton_impl = None

pytestmark = pytest.mark.skipif(triton_impl is None, reason="Triton not available")


def test_available_corrint_backends_shape_and_keys():
    out = available_corrint_backends()
    assert set(out.keys()) == {"triton", "pytorch", "pytorch_fast"}
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


def test_triton_cross_same_tensor_forwards_seq_lens(monkeypatch):
    vecs = torch.randn(2, 8, 3)
    eps = torch.tensor([0.25, 1.5], dtype=torch.float32)
    seq_lens = torch.tensor([8, 5], dtype=torch.int32)
    recorded = {}

    def fake_batched_self(v, e, seq_lens=None):
        recorded["vecs_is_same"] = v is vecs
        recorded["eps_is_same"] = e is eps
        recorded["seq_lens"] = seq_lens
        return torch.zeros((2, 2), dtype=torch.int64)

    monkeypatch.setattr(triton_impl, "_batched_correlation_counts", fake_batched_self)
    out = triton_impl._batched_correlation_counts_cross(
        vecs,
        vecs,
        eps,
        seq_lens1=seq_lens,
    )
    assert out.shape == (2, 2)
    assert recorded["vecs_is_same"] is True
    assert recorded["eps_is_same"] is True
    assert torch.equal(recorded["seq_lens"], seq_lens)


def test_triton_cross_same_tensor_rejects_mismatched_seq_lens():
    vecs = torch.randn(2, 8, 3)
    eps = torch.tensor([0.25, 1.5], dtype=torch.float32)
    with pytest.raises(ValueError, match="must match"):
        _ = triton_impl._batched_correlation_counts_cross(
            vecs,
            vecs,
            eps,
            seq_lens1=torch.tensor([8, 5], dtype=torch.int32),
            seq_lens2=torch.tensor([8, 4], dtype=torch.int32),
        )


@pytest.mark.parametrize(
    "seq_lens,msg",
    [
        (torch.tensor([8], dtype=torch.int32), "shape"),
        (torch.tensor([8, -1], dtype=torch.int32), "non-negative"),
        (torch.tensor([8, 99], dtype=torch.int32), "max sequence length"),
        ([8, 5, 1], "shape"),
    ],
)
def test_triton_seq_lens_validation_errors(seq_lens, msg):
    vecs = torch.randn(2, 8, 3)
    eps = torch.tensor([0.25, 1.5], dtype=torch.float32)
    with pytest.raises(ValueError, match=msg):
        _ = triton_impl.correlation_counts(vecs, eps, seq_lens=seq_lens)


def test_triton_correlation_integral_self_uses_seq_lens_denominator(monkeypatch):
    vecs = torch.randn(2, 8, 3)
    eps = torch.tensor([0.25, 1.5], dtype=torch.float32)
    seq_lens = torch.tensor([8, 5], dtype=torch.int32)  # denom: 56, 20
    fake_counts = torch.tensor([[56, 28], [10, 5]], dtype=torch.int64)

    monkeypatch.setattr(triton_impl, "correlation_counts", lambda *args, **kwargs: fake_counts.clone())
    got = triton_impl.correlation_integral(vecs, eps, seq_lens=seq_lens)
    expected = fake_counts.to(torch.float32) / torch.tensor([[56.0], [20.0]], dtype=torch.float32)
    assert torch.allclose(got, expected)


def test_triton_correlation_integral_cross_uses_both_seq_lens_denominator(monkeypatch):
    vecs1 = torch.randn(2, 8, 3)
    vecs2 = torch.randn(2, 10, 3)
    eps = torch.tensor([0.25, 1.5], dtype=torch.float32)
    seq_lens = torch.tensor([8, 5], dtype=torch.int32)
    seq_lens_other = torch.tensor([10, 4], dtype=torch.int32)  # denom: 80, 20
    fake_counts = torch.tensor([[80, 40], [10, 5]], dtype=torch.int64)

    monkeypatch.setattr(triton_impl, "correlation_counts", lambda *args, **kwargs: fake_counts.clone())
    got = triton_impl.correlation_integral(
        vecs1,
        eps,
        vecs_other=vecs2,
        seq_lens=seq_lens,
        seq_lens_other=seq_lens_other,
    )
    expected = fake_counts.to(torch.float32) / torch.tensor([[80.0], [20.0]], dtype=torch.float32)
    assert torch.allclose(got, expected)


def test_triton_progressive_integral_masks_positions_beyond_seq_lens(monkeypatch):
    vecs = torch.randn(2, 8, 3)
    eps = torch.tensor([0.25, 1.5], dtype=torch.float32)
    seq_lens = torch.tensor([8, 5], dtype=torch.int32)
    fake_counts = torch.ones((2, 8, 2), dtype=torch.int64)

    monkeypatch.setattr(triton_impl, "progressive_correlation_counts", lambda *args, **kwargs: fake_counts.clone())
    got = triton_impl.progressive_correlation_integral(vecs, eps, seq_lens=seq_lens)
    assert torch.all(got[1, 5:, :] == 0)
    assert torch.all(got[0, :, :] > 0)


def test_triton_rejects_seq_lens_other_without_vecs_other():
    vecs = torch.randn(2, 8, 3)
    eps = torch.tensor([0.25, 1.5], dtype=torch.float32)
    with pytest.raises(ValueError, match="seq_lens_other is only valid"):
        _ = triton_impl.correlation_counts(
            vecs,
            eps,
            seq_lens_other=torch.tensor([8, 7], dtype=torch.int32),
        )

