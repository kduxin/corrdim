# CorrDim: Correlation Dimension for Language Models

A Python library for computing correlation dimension of autoregressive large language models, based on the research paper ["Correlation Dimension of Auto-Regressive Large Language Models"](https://arxiv.org/abs/2510.21258) (NeurIPS 2025).

## Overview

Correlation dimension is a fractal-geometric measure of self-similarity that quantifies the epistemological complexity of text as perceived by a language model. Unlike traditional evaluation metrics that focus on local prediction accuracy, correlation dimension captures long-range structural complexity and can detect various forms of degeneration in generated text.

**Key Features:**
- **Bridges local and global perspectives**: Captures long-range structural complexity beyond local prediction accuracy (perplexity)
- **Detects various forms of degeneration**: Identifies repetition, incoherence, and blandness in generated text—issues that perplexity alone cannot reliably capture
- **Reveals training dynamics**: Uncovers three distinct phases during LLM pretraining (short-range learning, long-range emergence, generalization) that remain hidden to perplexity-based metrics
- **Computationally efficient**: Requires only next-token log-probability vectors at inference time, making it easy to integrate into existing workflows
- **Model-agnostic**: Works with any autoregressive language model (Transformer, Mamba, etc.) and is robust to quantization

## Installation

**Requirements**: Python >= 3.10

### Install matrix

```bash
# Default install (includes Triton dependency)
pip install corrdim
```

```bash
# CPU-only / no Triton dependency
pip install "corrdim[no-triton]"
```

```bash
# Development install
pip install "corrdim[dev]"
```

### Notes on Triton and backends

- Default installation includes Triton so CUDA users can use the fastest backend when available.
- `corrdim[no-triton]` avoids Triton installation and uses PyTorch backends.
- Backend selection is automatic by default:
  - if CUDA + Triton are available: `triton`
  - otherwise: `pytorch`
- You can override backend at runtime:

```bash
export CORRDIM_CORRINT_BACKEND=pytorch
```

(`CORRDIM_BACKEND` is still accepted for backward compatibility.)

or in code:

```python
import corrdim
corrdim.set_corrint_backend("pytorch")
```

## Quick Start

### Command-Line Interface

```bash
corrdim measure-text \
  --file data/sep60/newton-philosophy.txt \
  --model Qwen/Qwen2.5-1.5B
```

If you prefer running without installing globally:

```bash
uv run corrdim measure-text \
  --file data/sep60/newton-philosophy.txt \
  --model Qwen/Qwen2.5-1.5B
```

### Development install (repository)

If you are developing from source:

```bash
git clone https://github.com/kduxin/corrdim.git
cd corrdim
uv sync
```

### High-Level Interface

```python
import torch
import corrdim

result = corrdim.measure_text(
    "Your text here...",
    model="Qwen/Qwen2.5-1.5B",
    precision=torch.float16,
    truncation_tokens=10000,
    dim_reduction=8192,
)
print(f"Correlation dimension: {result.corrdim:.2f}")
```

Text APIs default to dimensionality reduction with `dim_reduction=8192`.  
To disable reduction explicitly, pass `dim_reduction=None`.

### Low-Level Interface (vectors / batch / text)

```python
import torch
import transformers
import corrdim

# Get log-probabilities from your model
model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.float16).to("cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
inputs = tokenizer(text, return_tensors="pt").to("cuda")
logprobs = model(**inputs).logits[0].log_softmax(-1)

# Compute correlation integral and dimension
curve = corrdim.curve_from_vectors(
    logprobs,
    num_epsilon=1024,
)
result = corrdim.estimate_dimension_from_curve(curve)
print(f"Correlation dimension: {result.corrdim:.2f}")

# Batch vectors
vecs_batch = torch.randn(4, 150, 512, device="cuda")
curves = corrdim.curve_from_vectors_batch(vecs_batch)
results = corrdim.estimate_dimension_from_curves(curves)

# Text and text batch paths (internally compute log-probs)
curve_text = corrdim.curve_from_text("hello world", model="Qwen/Qwen2.5-1.5B")
curve_texts = corrdim.curve_from_texts(["a", "b"], model="Qwen/Qwen2.5-1.5B")
prog = corrdim.progressive_curve_from_vectors(logprobs)
prog_batch = corrdim.progressive_curve_from_vectors_batch(vecs_batch)
prog_text = corrdim.progressive_curve_from_text("hello", model="Qwen/Qwen2.5-1.5B")
prog_texts = corrdim.progressive_curve_from_texts(["a", "b"], model="Qwen/Qwen2.5-1.5B")
```

### Backend control

```python
corrdim.set_corrint_backend("triton")
```

Available backend names:
- `"triton"`
- `"pytorch"`
- `"pytorch_fast"`
- `"auto"` (default)

`*_from_text*` and `measure_*` use a process-wide LRU model cache (size 2) to avoid repeated model loading.
You can clear it manually:

```python
corrdim.clear_model_cache()
```

### From-Curve Interface

```python
import corrdim

curve = corrdim.CurveResult(
    sequence_length=1500,
    epsilons=epsilons_np,
    corrints=corrints_np,
)
result = corrdim.estimate_dimension_from_curve(curve)
```

See [examples/basic_usage.ipynb](examples/basic_usage.ipynb) for more detailed examples.

## CLI Usage

The package exposes a `corrdim` CLI with three subcommands:

```bash
# 1) Text -> correlation dimension
corrdim measure-text \
  --file data/sep60/chaos.txt \
  --model Qwen/Qwen2.5-1.5B \
  --dim-reduction 8192 \
  --dtype float16

# 2) Text -> curve JSON
corrdim curve-from-text \
  --file data/sep60/chaos.txt \
  --model Qwen/Qwen2.5-1.5B \
  --output curve.json

# 3) Curve JSON -> correlation dimension
corrdim estimate-dimension \
  --curve-json curve.json
```

`--dim-reduction none` can be used to disable dimensionality reduction.

Useful options:
- `--backend triton|pytorch|pytorch_fast|auto`
- `--stride <positive-int>` (default: `1`)
- `--context-length <int>`
- `--num-epsilon <int>`

## API Reference

### High-Level

- `measure_text(text, model, ...)` → `DimensionResult`
- `measure_texts(texts, model, ...)` → `list[DimensionResult]`

### Low-Level

- `curve_from_vectors(vectors, ...)` / `curve_from_vectors_batch(vectors_batch, ...)`
- `curve_from_text(text, model, ...)` / `curve_from_texts(texts, model, ...)`
- `progressive_curve_from_vectors(...)` / `progressive_curve_from_vectors_batch(...)`
- `progressive_curve_from_text(...)` / `progressive_curve_from_texts(...)`

### From-Curve

- `estimate_dimension_from_curve(curve, ...)` → `DimensionResult`
- `estimate_dimension_from_curves(curves, ...)` → `list[DimensionResult]`

### Backend Utilities

- `correlation_integral(vecs, epsilons, backend=...)`
- `set_corrint_backend(backend)` (`"triton"`, `"pytorch"`, `"pytorch_fast"`)

## Citation

```bibtex
@inproceedings{du2025correlation,
  title={Correlation Dimension of Auto-Regressive Large Language Models},
  author={Du, Xin and Tanaka-Ishii, Kumiko},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
  arxiv={2510.21258}
}
```

## Links

- **Paper**: [arXiv:2510.21258](https://arxiv.org/abs/2510.21258)
- **Repository**: [https://github.com/kduxin/corrdim](https://github.com/kduxin/corrdim)

## License

MIT License
