# Concepts

## What CorrDim measures

Given a text and an autoregressive language model, CorrDim measures the text's **global structural complexity** as perceived by that model.

At a high level:

- repetitive or degenerate text tends to have a lower correlation dimension
- ordinary fluent text tends to have a higher dimension
- richer long-range structure can produce an even higher dimension

CorrDim is therefore best treated as a sequence-level geometric signal, not as a replacement for perplexity.

## How the pipeline works

CorrDim typically follows four steps:

1. Convert text into a sequence of next-token log-probability vectors.
2. Optionally reduce the vocabulary dimension.
3. Compute a correlation-integral curve over a range of epsilon thresholds.
4. Fit the slope in log-log space to estimate the correlation dimension.

In Python, these stages map roughly to:

- `curve_from_text(...)` or `curve_from_vectors(...)` for curve construction
- `estimate_dimension_from_curve(...)` for slope fitting
- `measure_text(...)` when you want both steps wrapped into one call
- `measure_text_progressive(...)` when you want fitted dimensions at subsampled prefix lengths after a single progressive curve pass

## Backend model

CorrDim exposes multiple backends for correlation-integral computation:

- `cuda`: custom CUDA extension
- `triton`: Triton kernels
- `pytorch`: pure PyTorch implementation
- `pytorch_fast`: PyTorch variant optimized for distance computation
- `auto`: resolve automatically, preferring `cuda` when available and otherwise `pytorch`

You can select the backend with an environment variable:

```bash
export CORRDIM_CORRINT_BACKEND=pytorch
```

Or in Python:

```python
import corrdim

resolved = corrdim.set_corrint_backend("auto")
print("Using backend:", resolved)
print(corrdim.available_corrint_backends())
```

If you do not set anything, CorrDim defaults to `triton`.

## API layers

The library is intentionally split into layers:

- high-level API: `measure_text`, `measure_texts`, `measure_text_progressive`
- curve API: `curve_from_text`, `curve_from_texts`, `curve_from_vectors`
- progressive API: `progressive_curve_from_text`, `progressive_curve_from_vectors`; fitted dimensions along prefixes use `measure_text_progressive` → `ProgressiveDimensionResult`
- raw backend API: `correlation_counts`, `correlation_integral`, `progressive_correlation_integral`

Use the highest layer that still gives you the outputs you need.
