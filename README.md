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

**Requirements**: Python >= 3.11

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. If you don't have uv installed:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

Then initialize and install the package:

```bash
git clone https://github.com/kduxin/corrdim.git
cd corrdim
uv sync

# Optional: AWQ-quantized model support
uv add autoawq
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

See [examples/basic_usage.ipynb](examples/basic_usage.ipynb) for more detailed examples.

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
