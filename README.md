# CorrDim: Correlation Dimension for Language Models

CorrDim is a Python library for computing the **correlation dimension** of autoregressive language models from next-token log-probability vectors, based on the paper **["Correlation Dimension of Auto-Regressive Large Language Models"](https://arxiv.org/abs/2510.21258)** (NeurIPS 2025).

## Documentation

Full documentation is available at [corrdim.readthedocs.io](https://corrdim.readthedocs.io).

Use the docs site for:

- installation details and backend notes
- the full Python API reference
- CLI documentation
- examples and usage patterns

## What CorrDim measures

Given a text and an autoregressive language model, CorrDim measures the text's **global structural complexity** as perceived by that model.

In practice:

- repetitive or degenerate text tends to have a lower correlation dimension
- ordinary fluent text tends to have a higher dimension
- richer long-range structure can produce an even higher dimension

CorrDim is complementary to local metrics such as perplexity: it focuses on **sequence-level geometry**, not just token-level prediction quality.

## How it works

At a high level, CorrDim:

1. converts text into a sequence of next-token log-probability vectors
2. optionally reduces the vocabulary dimension
3. computes a correlation-integral curve over epsilon thresholds
4. estimates the correlation dimension by fitting a line in log-log space

For the mathematical details, see the [paper](https://arxiv.org/abs/2510.21258).

## Installation

CorrDim requires Python 3.10 or newer and PyTorch >= 2.3.

**1. Install PyTorch** (choose the right build for your hardware):

```bash
# Auto-detect (recommended): CUDA, ROCm, MPS, or CPU
uv pip install torch --torch-backend=auto

# Or use pip with the PyTorch CUDA index
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**2. Install CorrDim**:

```bash
pip install corrdim
```

For local development:

```bash
pip install "corrdim[dev,docs]"
```

To compile the CUDA extension during installation:

```bash
CORRDIM_BUILD_CUDA=1 pip install .
```

## Quick start

```python
import torch
import corrdim

result = corrdim.measure_text(
    "Your text here...",
    model="Qwen/Qwen2.5-1.5B",
    precision=torch.float16,
)

print("corrdim:", result.corrdim)
print("fit_r2:", result.fit_r2)
print("linear_region_bounds:", result.linear_region_bounds)
```

For batched input:

```python
import torch
import corrdim

results = corrdim.measure_texts(
    [
        "Short sample A...",
        "Short sample B...",
    ],
    model="Qwen/Qwen2.5-1.5B",
    precision=torch.float16,
)

for result in results:
    print(result.corrdim, result.fit_r2)
```

### Progressive dimension along the sequence

To fit correlation dimension at multiple prefix lengths without re-running the model for each prefix, use `measure_text_progressive`. It calls `progressive_curve_from_text` once, then subsamples prefix indices:

- `skip_prefix_tokens`: first prefix index to include (shorter prefixes are skipped)
- `measure_every_tokens`: stride between measured indices, or `None` (default) to choose from length: fewer than 100 tokens → `1`, fewer than 1000 → `10`, otherwise `100`

The return value is a `ProgressiveDimensionResult`: `by_prefix` maps prefix index to a full `DimensionResult`; `corrdims` maps index to the fitted scalar only.

```python
import torch
import corrdim

prog_dims = corrdim.measure_text_progressive(
    long_text,
    model="Qwen/Qwen2.5-1.5B",
    precision=torch.float16,
    skip_prefix_tokens=100,
)

for prefix_len, d in sorted(prog_dims.corrdims.items()):
    print(prefix_len, d)
```

## API overview

The most important entry points are:

- `measure_text` / `measure_texts` for end-to-end text measurement
- `measure_text_progressive` for multiple fitted dimensions along sequence prefixes (one model pass)
- `curve_from_text` / `curve_from_vectors` when you want the curve first
- `estimate_dimension_from_curve` when you already have saved curve data
- `progressive_curve_from_text` for prefix-wise analysis
- `correlation_integral` and related functions for lower-level tensor workflows

For full API details, signatures, return types, and backend behavior, see the [documentation site](https://corrdim.readthedocs.io).

## CLI

CorrDim includes a `corrdim` command-line interface:

```bash
corrdim measure-text \
  --file data/sep60/chaos.txt \
  --model Qwen/Qwen2.5-1.5B
```

Additional CLI commands and options are documented at [corrdim.readthedocs.io](https://corrdim.readthedocs.io).

## Backends

CorrDim supports multiple backends for correlation-integral computation:

- `cuda`
- `triton`
- `pytorch`
- `pytorch_fast`
- `auto`

Set the default backend with:

```bash
export CORRDIM_CORRINT_BACKEND=pytorch
```

Or in Python:

```python
import corrdim

print(corrdim.set_corrint_backend("auto"))
print(corrdim.available_corrint_backends())
```

## Tips for low-VRAM systems

If you run into out-of-memory errors, reduce `block_size` (default 512) to lower the peak memory usage during correlation-integral computation:

```python
result = corrdim.measure_text(
    text,
    model="Qwen/Qwen2.5-1.5B",
    block_size=128,
)
```

You can also set `forward_chunk_size` to control how many tokens are processed per forward pass (reduce this value, e.g. 128, on systems with limited VRAM):

```python
result = corrdim.measure_text(
    text,
    model="Qwen/Qwen2.5-1.5B",
    block_size=128,
    forward_chunk_size=128,
)
```

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

- Documentation: https://corrdim.readthedocs.io
- Paper: https://arxiv.org/abs/2510.21258
- Repository: https://github.com/kduxin/corrdim

## License

MIT License
