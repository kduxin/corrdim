import numpy as np
import pytest
import torch

import corrdim
from corrdim import low_level
from corrdim.models import LanguageModelWrapper


class DummyWrapper(LanguageModelWrapper):
    def __init__(self, seq_len: int = 160, vocab: int = 48, tokenizer=None):
        self.seq_len = seq_len
        self.vocab = vocab
        self.tokenizer = tokenizer

    def get_log_probabilities(
        self,
        text: str,
        context_length=None,
        dim_reduction=None,
        stride: int = 1,
        show_progress: bool = False,
    ):
        _ = (text, context_length, dim_reduction, stride, show_progress)
        return torch.randn(self.seq_len, self.vocab, dtype=torch.float32)


class VaryingLenWrapper(LanguageModelWrapper):
    def __init__(self, base_len: int = 120, vocab: int = 24):
        self.base_len = base_len
        self.vocab = vocab
        self.tokenizer = None
        self.calls: list[str] = []

    def get_log_probabilities(
        self,
        text: str,
        context_length=None,
        dim_reduction=None,
        stride: int = 1,
        show_progress: bool = False,
    ):
        _ = (context_length, dim_reduction, stride, show_progress)
        self.calls.append(text)
        seq_len = self.base_len + len(text)
        return torch.randn(seq_len, self.vocab, dtype=torch.float32)


def test_curve_from_vectors_rejects_nonfinite_vectors():
    vecs = torch.randn(150, 8, dtype=torch.float32)
    vecs[0, 0] = float("nan")

    with pytest.raises(ValueError, match="Found nan or inf in vectors"):
        _ = corrdim.curve_from_vectors(vecs, num_epsilon=8, backend="pytorch")


def test_curve_from_vectors_rejects_too_short_sequence():
    vecs = torch.randn(100, 8, dtype=torch.float32)  # must be >100

    with pytest.raises(ValueError, match="The sequence length is too short"):
        _ = corrdim.curve_from_vectors(vecs, num_epsilon=8, backend="pytorch")


def test_curve_from_vectors_batch_requires_dim_3():
    vecs = torch.randn(3, 5)  # dim=2
    with pytest.raises(ValueError, match="vectors_batch must have shape"):
        _ = corrdim.curve_from_vectors_batch(vecs, num_epsilon=8, backend="pytorch")


def test_model_cache_key_includes_kwargs_and_tokenizer_identity(monkeypatch):
    corrdim.clear_model_cache()

    created = []

    def _fake_create_model_wrapper(model_name, tokenizer=None, device=None, **kwargs):
        created.append((model_name, id(tokenizer) if tokenizer is not None else None, device, dict(kwargs)))
        return DummyWrapper(seq_len=110, vocab=16)

    monkeypatch.setattr(low_level, "create_model_wrapper", _fake_create_model_wrapper)

    tokenizer1 = object()
    tokenizer2 = object()

    corrdim.curve_from_text(
        "hello",
        model="m-1",
        tokenizer=tokenizer1,
        device="cpu",
        foo=1,
        num_epsilon=8,
        backend="pytorch",
    )
    # Cache hit: same model/tokenizer/device/kwargs
    corrdim.curve_from_text(
        "hello",
        model="m-1",
        tokenizer=tokenizer1,
        device="cpu",
        foo=1,
        num_epsilon=8,
        backend="pytorch",
    )
    # Cache miss: kwargs changed
    corrdim.curve_from_text(
        "hello",
        model="m-1",
        tokenizer=tokenizer1,
        device="cpu",
        foo=2,
        num_epsilon=8,
        backend="pytorch",
    )
    # Cache miss: tokenizer identity changed
    corrdim.curve_from_text(
        "hello",
        model="m-1",
        tokenizer=tokenizer2,
        device="cpu",
        foo=1,
        num_epsilon=8,
        backend="pytorch",
    )

    assert len(created) == 3


def test_curve_from_texts_batch_size_fallback_handles_varying_lengths():
    wrapper = VaryingLenWrapper(base_len=120, vocab=24)
    texts = ["a", "medium_text", "x" * 40, "z" * 5, "q" * 17]
    curves = corrdim.curve_from_texts(texts, model=wrapper, batch_size=2, num_epsilon=16, backend="pytorch")

    assert wrapper.calls == texts
    assert len(curves) == len(texts)
    expected_lengths = [120 + len(t) for t in texts]
    assert [c.sequence_length for c in curves] == expected_lengths

