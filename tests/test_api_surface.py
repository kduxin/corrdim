import numpy as np
import torch

import corrdim
from corrdim import low_level
from corrdim.models import LanguageModelWrapper


class DummyWrapper(LanguageModelWrapper):
    def __init__(self, seq_len=150, vocab=64):
        self.seq_len = seq_len
        self.vocab = vocab
        self.tokenizer = None

    def get_log_probabilities(
        self,
        text: str,
        context_length=None,
        dim_reduction=None,
        stride=1,
        show_progress=False,
    ):
        _ = (text, context_length, dim_reduction, stride, show_progress)
        return torch.randn(self.seq_len, self.vocab)


class TrackingWrapper(LanguageModelWrapper):
    def __init__(self, seq_len=150, vocab=64):
        self.seq_len = seq_len
        self.vocab = vocab
        self.tokenizer = None
        self.last_dim_reduction = "unset"

    def get_log_probabilities(
        self,
        text: str,
        context_length=None,
        dim_reduction=None,
        stride=1,
        show_progress=False,
    ):
        _ = (text, context_length, stride, show_progress)
        self.last_dim_reduction = dim_reduction
        return torch.randn(self.seq_len, self.vocab)


def setup_module(module):
    _ = module
    corrdim.set_corrint_backend("pytorch")


def test_curve_from_vectors_and_dimension():
    vectors = torch.randn(150, 32)
    curve = corrdim.curve_from_vectors(vectors, num_epsilon=64)
    result = corrdim.estimate_dimension_from_curve(curve)
    assert curve.sequence_length == 150
    assert len(curve.epsilons) == len(curve.corrints)
    assert isinstance(result.corrdim, float)


def test_curve_from_vectors_batch():
    vectors_batch = torch.randn(3, 150, 16)
    curves = corrdim.curve_from_vectors_batch(vectors_batch, num_epsilon=32)
    assert len(curves) == 3
    assert all(curve.sequence_length == 150 for curve in curves)


def test_text_apis_with_wrapper():
    wrapper = DummyWrapper(seq_len=160, vocab=48)
    curve = corrdim.curve_from_text("hello", model=wrapper, num_epsilon=32)
    curves = corrdim.curve_from_texts(["a", "b"], model=wrapper, num_epsilon=32)
    dim = corrdim.measure_text("hello", model=wrapper, num_epsilon=32)
    dims = corrdim.measure_texts(["a", "b"], model=wrapper, num_epsilon=32)
    assert curve.sequence_length == 160
    assert len(curves) == 2
    assert dim.sequence_length == 160
    assert len(dims) == 2


def test_progressive_variants():
    vectors = torch.randn(150, 20)
    prog = corrdim.progressive_curve_from_vectors(vectors, num_epsilon=16)
    assert prog.corrints_progressive.shape[0] == 150
    assert prog.corrints_progressive.shape[1] == len(prog.epsilons)


def test_text_default_dim_reduction_and_none():
    wrapper = TrackingWrapper(seq_len=160, vocab=48)

    corrdim.curve_from_text("hello", model=wrapper, num_epsilon=16)
    assert wrapper.last_dim_reduction == 8192

    corrdim.curve_from_text("hello", model=wrapper, dim_reduction=None, num_epsilon=16)
    assert wrapper.last_dim_reduction is None


def test_model_cache_lru2(monkeypatch):
    corrdim.clear_model_cache()
    built_models = []

    def _fake_create_model_wrapper(model_name, tokenizer=None, device=None, **kwargs):
        _ = (tokenizer, device, kwargs)
        built_models.append(model_name)
        return DummyWrapper(seq_len=160, vocab=48)

    monkeypatch.setattr(low_level, "create_model_wrapper", _fake_create_model_wrapper)

    corrdim.curve_from_text("a", model="model-1", num_epsilon=16)
    corrdim.curve_from_text("b", model="model-1", num_epsilon=16)
    corrdim.curve_from_text("c", model="model-2", num_epsilon=16)
    corrdim.curve_from_text("d", model="model-3", num_epsilon=16)
    corrdim.curve_from_text("e", model="model-1", num_epsilon=16)

    # Expect: model-1 cached hit once, then evicted after model-2/model-3 inserted, rebuilt at end.
    assert built_models == ["model-1", "model-2", "model-3", "model-1"]
