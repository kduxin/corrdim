# Installation

CorrDim requires Python 3.10 or newer.

## Package install

```bash
pip install corrdim
```

### Linux GPU (CUDA PyTorch)

By default, PyPI distributes CPU-only PyTorch on Linux. If you have an NVIDIA GPU, install CUDA PyTorch first. Choose based on your driver version:

| CUDA version | Min driver | Install command |
|---|---|---|
| cu126 (default) | ≥ 560 | `pip install torch --index-url https://download.pytorch.org/whl/cu126` |
| cu130 | ≥ 580 | `pip install torch --index-url https://download.pytorch.org/whl/cu130` |

If using `uv`, the cu126 index is configured by default in `pyproject.toml`. To switch to cu130, change the torch source to `pytorch-cu130`.

If you want to avoid installing Triton:

```bash
pip install "corrdim[no-triton]"
```

If you are developing locally:

```bash
pip install "corrdim[dev,docs]"
```

## Install from source

From the repository root:

```bash
pip install .
```

## Build the docs locally

The Read the Docs configuration uses a lightweight docs environment:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser after the build completes.
