# CorrDim: Correlation Dimension for Language Models

A Python library for computing correlation dimension of autoregressive large language models, based on the research paper ["Correlation Dimension of Auto-Regressive Large Language Models"](https://arxiv.org/abs/2510.21258) (NeurIPS 2025).

## Overview

Correlation dimension is a fractal-geometric measure of self-similarity that quantifies the epistemological complexity of text as perceived by a language model. Unlike traditional evaluation metrics that focus on local prediction accuracy, correlation dimension captures long-range structural complexity and can detect various forms of degeneration in generated text.

**Key Features:**
- Efficient computation with Triton support
- Works with any autoregressive language model (Transformer, Mamba, etc.)
- Robust to model quantization down to 4-bit precision
- Reveals distinct phases during language model pretraining

## Installation

```bash
git clone https://github.com/kduxin/corrdim.git
cd corrdim
uv pip install -e .

# Optional: AWQ-quantized model support
uv pip install -e ".[awq]"

# Optional: Demo applications
uv pip install -e ".[demo]"
```

## Quick Start

### High-Level Interface

```python
import torch
import corrdim

calculator = corrdim.CorrelationDimensionCalculator(
    "Qwen/Qwen2.5-1.5B", 
    device="cuda", 
    dtype=torch.float16
)

result = calculator("Your text here...", dim_reduction=8192)
print(f"Correlation dimension: {result.corrdim:.2f}")
```

### Low-Level Interface

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
epsilons = torch.logspace(-20, 20, 10000, device="cuda")
corrint = corrdim.correlation_integral(logprobs, epsilons)
result = corrdim.CorrelationDimensionCalculator.compute_correlation_dimension_from_curve(
    epsilons=epsilons.cpu().numpy(),
    corrints=corrint.cpu().numpy(),
    sequence_length=logprobs.shape[0]
)
print(f"Correlation dimension: {result.corrdim:.2f}")
```

See [examples/sep60.ipynb](examples/sep60.ipynb) for more detailed examples.

## API Reference

### `CorrelationDimensionCalculator`

Main class for computing correlation dimension.

- `__call__(text, ...)` → `CorrelationDimensionResult`: Compute correlation dimension
- `compute_correlation_integral_curve(text, ...)` → `CorrelationIntegralResult`: Get full curve
- `get_log_probability(text, ...)` → `torch.Tensor`: Extract log-probability vectors

### `CorrelationDimensionResult`

- `corrdim` (float): Correlation dimension value
- `fit_r2` (float): R² score of the linear fit
- `epsilons`, `corrints` (np.ndarray): Full curve data
- `epsilons_linear_region`, `corrints_linear_region` (np.ndarray): Linear region data
- `linear_region_bounds` (Tuple[float, float]): Bounds of linear region
- `sequence_length` (int): Input sequence length

### Utility Functions

- `correlation_integral(vecs, epsilons)`: Compute correlation integral
- `set_corrint_backend(backend)`: Set backend ("triton", "pytorch", or "pytorch_fast")

## Requirements

Python >= 3.8, PyTorch >= 2.0.0, Transformers >= 4.30.0, NumPy >= 1.21.0, scikit-learn >= 1.0.0, tqdm >= 4.62.0

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
