import pytest
import torch
from types import SimpleNamespace

import corrdim
import corrdim.high_level as high_level
from corrdim.models import LanguageModelWrapper


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        # Tokens are just [0..len(text)-1] so truncation changes the decoded string deterministically.
        _ = add_special_tokens
        return list(range(len(text)))

    def decode(self, tokens, skip_special_tokens: bool = False):
        _ = skip_special_tokens
        return "|".join(str(t) for t in tokens)


class RecordingTextWrapper(LanguageModelWrapper):
    def __init__(self, seq_len: int = 160, vocab: int = 48, tokenizer=None):
        self.seq_len = seq_len
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.last_text = None
        self.last_dim_reduction = "unset"

    def get_log_probabilities(
        self,
        text: str,
        context_length=None,
        dim_reduction=None,
        stride: int = 1,
        show_progress: bool = False,
    ):
        _ = (context_length, stride, show_progress)
        self.last_text = text
        self.last_dim_reduction = dim_reduction
        return torch.randn(self.seq_len, self.vocab, dtype=torch.float32)


class NoTokenizerWrapper(LanguageModelWrapper):
    def __init__(self, seq_len: int = 160, vocab: int = 48):
        self.seq_len = seq_len
        self.vocab = vocab

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


def test_measure_text_truncation_by_tokens_uses_tokenizer_decode(monkeypatch):
    # Make sure we stay on CPU backend for speed/reliability.
    corrdim.set_corrint_backend("pytorch")

    tokenizer = DummyTokenizer()
    wrapper = RecordingTextWrapper(seq_len=160, vocab=32, tokenizer=tokenizer)

    _ = corrdim.measure_text(
        "abcd",
        model=wrapper,
        tokenizer=None,
        truncation_tokens=2,
        dim_reduction=8,
        num_epsilon=32,
        precision=torch.float32,
        backend="pytorch",
    )

    assert wrapper.last_text == "0|1"


def test_measure_text_truncation_tokens_must_be_positive():
    corrdim.set_corrint_backend("pytorch")
    wrapper = RecordingTextWrapper(seq_len=160, vocab=32, tokenizer=DummyTokenizer())

    with pytest.raises(ValueError, match="truncation_tokens must be a positive integer"):
        _ = corrdim.measure_text(
            "abcd",
            model=wrapper,
            tokenizer=None,
            truncation_tokens=0,
            dim_reduction=8,
            num_epsilon=8,
            precision=torch.float32,
            backend="pytorch",
        )


def test_measure_text_truncation_requires_tokenizer_or_model_tokenizer():
    corrdim.set_corrint_backend("pytorch")
    wrapper = NoTokenizerWrapper(seq_len=160, vocab=32)

    with pytest.raises(ValueError, match="Cannot truncate by tokens without a tokenizer"):
        _ = corrdim.measure_text(
            "abcd",
            model=wrapper,
            tokenizer=None,
            truncation_tokens=2,
            dim_reduction=8,
            num_epsilon=8,
            precision=torch.float32,
            backend="pytorch",
        )


def test_progressive_curve_from_text_shapes():
    corrdim.set_corrint_backend("pytorch")
    wrapper = RecordingTextWrapper(seq_len=110, vocab=24, tokenizer=DummyTokenizer())

    prog = corrdim.progressive_curve_from_text(
        "hello",
        model=wrapper,
        tokenizer=None,
        dim_reduction=8,
        num_epsilon=8,
        precision=torch.float32,
        backend="pytorch",
    )

    assert prog.sequence_length == 110
    assert len(prog.epsilons) == 8
    assert prog.corrints_progressive.shape == (110, 8)

    progs = corrdim.progressive_curve_from_texts(
        ["a", "b"],
        model=wrapper,
        tokenizer=None,
        dim_reduction=8,
        num_epsilon=6,
        precision=torch.float32,
        backend="pytorch",
    )
    assert len(progs) == 2
    assert progs[0].corrints_progressive.shape == (110, 6)


def test_progressive_curve_from_vectors_batch_shapes_and_validation():
    corrdim.set_corrint_backend("pytorch")
    vectors_batch = torch.randn(2, 110, 12, dtype=torch.float32)
    progs = corrdim.progressive_curve_from_vectors_batch(
        vectors_batch,
        num_epsilon=6,
        backend="pytorch",
    )
    assert len(progs) == 2
    assert all(p.sequence_length == 110 for p in progs)
    assert all(p.corrints_progressive.shape == (110, 6) for p in progs)

    with pytest.raises(ValueError, match="vectors_batch must have shape"):
        _ = corrdim.progressive_curve_from_vectors_batch(vectors_batch[0], num_epsilon=4, backend="pytorch")


def test_measure_text_defaults_enable_auto_corr_range(monkeypatch):
    recorded = {}
    fake_curve = SimpleNamespace()

    def fake_curve_from_text(**kwargs):
        recorded["epsilon_range"] = kwargs["epsilon_range"]
        return fake_curve

    def fake_estimate_dimension_from_curve(curve, correlation_integral_range, epsilon_range):
        recorded["curve"] = curve
        recorded["correlation_integral_range"] = correlation_integral_range
        recorded["estimate_epsilon_range"] = epsilon_range
        return "ok"

    monkeypatch.setattr(high_level, "curve_from_text", fake_curve_from_text)
    monkeypatch.setattr(high_level, "estimate_dimension_from_curve", fake_estimate_dimension_from_curve)

    out = high_level.measure_text("abc", model="dummy")

    assert out == "ok"
    assert recorded["curve"] is fake_curve
    assert recorded["epsilon_range"] == high_level.DEFAULT_EPSILON_RANGE
    assert recorded["correlation_integral_range"] is None
    assert recorded["estimate_epsilon_range"] is None


def test_measure_text_none_corr_range_with_explicit_epsilon_stays_none(monkeypatch):
    recorded = {}
    fake_curve = SimpleNamespace()
    explicit_epsilon_range = (1e-6, 1e-2)

    def fake_curve_from_text(**kwargs):
        recorded["epsilon_range"] = kwargs["epsilon_range"]
        return fake_curve

    def fake_estimate_dimension_from_curve(curve, correlation_integral_range, epsilon_range):
        recorded["curve"] = curve
        recorded["correlation_integral_range"] = correlation_integral_range
        recorded["estimate_epsilon_range"] = epsilon_range
        return "ok"

    monkeypatch.setattr(high_level, "curve_from_text", fake_curve_from_text)
    monkeypatch.setattr(high_level, "estimate_dimension_from_curve", fake_estimate_dimension_from_curve)

    out = high_level.measure_text(
        "abc",
        model="dummy",
        correlation_integral_range=None,
        epsilon_range=explicit_epsilon_range,
    )

    assert out == "ok"
    assert recorded["curve"] is fake_curve
    assert recorded["epsilon_range"] == explicit_epsilon_range
    assert recorded["correlation_integral_range"] is None
    assert recorded["estimate_epsilon_range"] == explicit_epsilon_range


def test_measure_text_explicit_epsilon_range_disables_auto_corr_range(monkeypatch):
    recorded = {}
    fake_curve = SimpleNamespace()
    explicit_epsilon_range = (1e-6, 1e-2)

    def fake_curve_from_text(**kwargs):
        recorded["epsilon_range"] = kwargs["epsilon_range"]
        return fake_curve

    def fake_estimate_dimension_from_curve(curve, correlation_integral_range, epsilon_range):
        recorded["curve"] = curve
        recorded["correlation_integral_range"] = correlation_integral_range
        recorded["estimate_epsilon_range"] = epsilon_range
        return "ok"

    monkeypatch.setattr(high_level, "curve_from_text", fake_curve_from_text)
    monkeypatch.setattr(high_level, "estimate_dimension_from_curve", fake_estimate_dimension_from_curve)

    out = high_level.measure_text("abc", model="dummy", epsilon_range=explicit_epsilon_range)

    assert out == "ok"
    assert recorded["curve"] is fake_curve
    assert recorded["epsilon_range"] == explicit_epsilon_range
    assert recorded["correlation_integral_range"] is None
    assert recorded["estimate_epsilon_range"] == explicit_epsilon_range


def test_measure_texts_explicit_epsilon_range_disables_auto_corr_range(monkeypatch):
    recorded = {"correlation_integral_ranges": [], "estimate_epsilon_ranges": []}
    fake_curves = [SimpleNamespace(name="c1"), SimpleNamespace(name="c2")]
    explicit_epsilon_range = (1e-5, 1e-1)

    def fake_curve_from_texts(**kwargs):
        recorded["epsilon_range"] = kwargs["epsilon_range"]
        return fake_curves

    def fake_estimate_dimension_from_curve(curve, correlation_integral_range, epsilon_range):
        recorded["correlation_integral_ranges"].append(correlation_integral_range)
        recorded["estimate_epsilon_ranges"].append(epsilon_range)
        return curve.name

    monkeypatch.setattr(high_level, "curve_from_texts", fake_curve_from_texts)
    monkeypatch.setattr(high_level, "estimate_dimension_from_curve", fake_estimate_dimension_from_curve)

    out = high_level.measure_texts(["a", "b"], model="dummy", epsilon_range=explicit_epsilon_range)

    assert out == ["c1", "c2"]
    assert recorded["epsilon_range"] == explicit_epsilon_range
    assert recorded["correlation_integral_ranges"] == [None, None]
    assert recorded["estimate_epsilon_ranges"] == [explicit_epsilon_range, explicit_epsilon_range]

