import math

import numpy as np
import pytest
import torch
from types import SimpleNamespace

import corrdim
import corrdim.high_level as high_level
from corrdim.models import LanguageModelWrapper

# Progressive tests use at least 100 tokens (tokenizer length and/or model seq_len).
_PROGRESSIVE_SEQ_LEN = 256
_PROGRESSIVE_TEXT = "t " * _PROGRESSIVE_SEQ_LEN


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
    wrapper = RecordingTextWrapper(seq_len=_PROGRESSIVE_SEQ_LEN, vocab=24, tokenizer=DummyTokenizer())

    prog = corrdim.progressive_curve_from_text(
        _PROGRESSIVE_TEXT,
        model=wrapper,
        tokenizer=None,
        dim_reduction=8,
        num_epsilon=8,
        precision=torch.float32,
        backend="pytorch",
    )

    assert prog.sequence_length == _PROGRESSIVE_SEQ_LEN
    assert len(prog.epsilons) == 8
    assert prog.corrints_progressive.shape == (_PROGRESSIVE_SEQ_LEN, 8)

    progs = corrdim.progressive_curve_from_texts(
        [_PROGRESSIVE_TEXT, "u" * _PROGRESSIVE_SEQ_LEN],
        model=wrapper,
        tokenizer=None,
        dim_reduction=8,
        num_epsilon=6,
        precision=torch.float32,
        backend="pytorch",
    )
    assert len(progs) == 2
    assert progs[0].corrints_progressive.shape == (_PROGRESSIVE_SEQ_LEN, 6)


def test_measure_text_progressive_keys_and_finite_corrdim():
    corrdim.set_corrint_backend("pytorch")
    wrapper = RecordingTextWrapper(seq_len=_PROGRESSIVE_SEQ_LEN, vocab=24, tokenizer=DummyTokenizer())

    skip, step = 10, 20
    out = corrdim.measure_text_progressive(
        _PROGRESSIVE_TEXT,
        model=wrapper,
        tokenizer=None,
        skip_prefix_tokens=skip,
        measure_every_tokens=step,
        dim_reduction=8,
        num_epsilon=8,
        precision=torch.float32,
        backend="pytorch",
    )
    assert out.sequence_length == _PROGRESSIVE_SEQ_LEN
    assert out.skip_prefix_tokens == skip
    assert out.measure_every_tokens == step
    assert out.epsilons.shape == (8,)
    assert set(out.by_prefix.keys()) == set(range(skip, _PROGRESSIVE_SEQ_LEN, step))
    assert out.corrdims == {i: out.by_prefix[i].corrdim for i in out.by_prefix}
    for res in out.by_prefix.values():
        assert res.sequence_length == _PROGRESSIVE_SEQ_LEN
        assert math.isfinite(res.corrdim)


def test_measure_text_progressive_validates_sampling_params():
    with pytest.raises(ValueError, match="skip_prefix_tokens"):
        high_level.measure_text_progressive(_PROGRESSIVE_TEXT, model="unused", skip_prefix_tokens=-1)
    with pytest.raises(ValueError, match="measure_every_tokens"):
        high_level.measure_text_progressive(_PROGRESSIVE_TEXT, model="unused", measure_every_tokens=0)


def test_default_measure_every_tokens_by_sequence_length():
    f = high_level._default_measure_every_tokens
    assert f(50) == 1
    assert f(99) == 1
    assert f(100) == 10
    assert f(500) == 10
    assert f(1000) == 100


@pytest.mark.parametrize(
    "seq_len,expected_step,skip",
    [
        (128, 10, 10),
        (500, 10, 10),
        (2000, 100, 100),
    ],
)
def test_measure_text_progressive_adaptive_measure_every(seq_len, expected_step, skip):
    """Full pipeline needs sequence length > 100 (``_make_epsilons``); skip early prefix rows that fit poorly."""
    torch.manual_seed(42)
    corrdim.set_corrint_backend("pytorch")
    wrapper = RecordingTextWrapper(seq_len=seq_len, vocab=24, tokenizer=DummyTokenizer())
    text = "x" * seq_len

    out = corrdim.measure_text_progressive(
        text,
        model=wrapper,
        tokenizer=None,
        skip_prefix_tokens=skip,
        measure_every_tokens=None,
        dim_reduction=8,
        num_epsilon=8,
        precision=torch.float32,
        backend="pytorch",
    )
    assert out.measure_every_tokens == expected_step
    assert set(out.by_prefix.keys()) == set(range(skip, seq_len, expected_step))


def test_progressive_curve_from_vectors_batch_shapes_and_validation():
    corrdim.set_corrint_backend("pytorch")
    vectors_batch = torch.randn(2, _PROGRESSIVE_SEQ_LEN, 12, dtype=torch.float32)
    progs = corrdim.progressive_curve_from_vectors_batch(
        vectors_batch,
        num_epsilon=6,
        backend="pytorch",
    )
    assert len(progs) == 2
    assert all(p.sequence_length == _PROGRESSIVE_SEQ_LEN for p in progs)
    assert all(p.corrints_progressive.shape == (_PROGRESSIVE_SEQ_LEN, 6) for p in progs)

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


def test_curve_from_texts_rejects_nonpositive_batch_size():
    corrdim.set_corrint_backend("pytorch")
    wrapper = RecordingTextWrapper(seq_len=120, vocab=16, tokenizer=DummyTokenizer())
    with pytest.raises(ValueError, match="batch_size"):
        corrdim.curve_from_texts(["x"], model=wrapper, batch_size=0, num_epsilon=8, precision=torch.float32, backend="pytorch")


def test_measure_texts_passes_batch_size_to_curve_from_texts(monkeypatch):
    recorded: dict = {}

    def fake_curve_from_texts(**kwargs):
        recorded["batch_size"] = kwargs.get("batch_size")
        return []

    monkeypatch.setattr(high_level, "curve_from_texts", fake_curve_from_texts)
    monkeypatch.setattr(high_level, "estimate_dimension_from_curve", lambda *a, **k: None)

    high_level.measure_texts(["only"], model="dummy", batch_size=3)
    assert recorded["batch_size"] == 3


def test_measure_texts_progressive_structure_and_batch_size(monkeypatch):
    corrdim.set_corrint_backend("pytorch")
    recorded: dict = {}

    def fake_progressive_curve_from_texts(**kwargs):
        texts = kwargs["texts"]
        recorded["batch_size"] = kwargs.get("batch_size")
        recorded["text_lengths"] = [len(t) for t in texts]
        eps = np.linspace(1e-3, 1e-1, 8)
        out = []
        for idx, _ in enumerate(texts):
            seq_len = _PROGRESSIVE_SEQ_LEN + idx * 17
            corrint = np.ones((seq_len, 8), dtype=np.float64) * 0.5
            out.append(
                corrdim.ProgressiveCurveResult(
                    sequence_length=seq_len,
                    epsilons=eps,
                    corrints_progressive=corrint,
                )
            )
        return out

    monkeypatch.setattr(high_level, "progressive_curve_from_texts", fake_progressive_curve_from_texts)

    t1 = "t " * _PROGRESSIVE_SEQ_LEN
    t2 = "u " * (_PROGRESSIVE_SEQ_LEN // 2)
    t3 = "v " * (_PROGRESSIVE_SEQ_LEN + 23)
    t4 = "w " * (_PROGRESSIVE_SEQ_LEN // 3)
    t5 = "x " * (_PROGRESSIVE_SEQ_LEN + 41)
    texts = [t1, t2, t3, t4, t5]
    outs = high_level.measure_texts_progressive(
        texts,
        model="dummy",
        skip_prefix_tokens=10,
        measure_every_tokens=20,
        dim_reduction=8,
        num_epsilon=8,
        precision=torch.float32,
        backend="pytorch",
        batch_size=2,
    )
    assert recorded["batch_size"] == 2
    assert len(set(recorded["text_lengths"])) > 1
    assert len(outs) == len(texts)
    expected_seq_lens = [_PROGRESSIVE_SEQ_LEN + i * 17 for i in range(len(texts))]
    for o, seq_len in zip(outs, expected_seq_lens):
        assert o.sequence_length == seq_len
        assert o.skip_prefix_tokens == 10
        assert o.measure_every_tokens == 20
        assert set(o.by_prefix.keys()) == set(range(10, seq_len, 20))


def test_measure_texts_progressive_validates_sampling_params():
    with pytest.raises(ValueError, match="skip_prefix_tokens"):
        high_level.measure_texts_progressive(["a", "b"], model="unused", skip_prefix_tokens=-1)
    with pytest.raises(ValueError, match="measure_every_tokens"):
        high_level.measure_texts_progressive(["a", "b"], model="unused", measure_every_tokens=0)

