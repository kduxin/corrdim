# Quick start

## Text to correlation dimension

This is the highest-level API:

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

## Get the curve first

If you want the full correlation-integral curve before fitting:

```python
import torch
import corrdim

curve = corrdim.curve_from_text(
    "Your text here...",
    model="Qwen/Qwen2.5-1.5B",
    precision=torch.float16,
)

result = corrdim.estimate_dimension_from_curve(curve)
print(curve.sequence_length)
print(curve.epsilons.shape)
print(curve.corrints.shape)
print(result.corrdim)
```

## Start from vectors

If you already have a sequence of log-probability vectors:

```python
import torch
import corrdim

vectors = torch.randn(200, 512, device="cuda")
curve = corrdim.curve_from_vectors(vectors, num_epsilon=256)
result = corrdim.estimate_dimension_from_curve(curve)
```

## Progressive analysis

To study how the curve evolves over prefixes of the sequence:

```python
import torch
import corrdim

progressive = corrdim.progressive_curve_from_text(
    "Your text here...",
    model="Qwen/Qwen2.5-1.5B",
    precision=torch.float16,
)

print(progressive.sequence_length)
print(progressive.epsilons.shape)
print(progressive.corrints_progressive.shape)
```

## Which function should I use?

- use `measure_text` / `measure_texts` for the final dimension directly
- use `curve_from_text` / `curve_from_vectors` when you want the raw curve
- use `progressive_curve_from_text` / `progressive_curve_from_vectors` for prefix-wise analysis
- use `correlation_integral` when you already have tensors and want raw backend outputs
