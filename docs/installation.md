# Installation

CorrDim requires Python 3.10 or newer.

## Package install

```bash
pip install corrdim
```

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

CUDA extension compilation is disabled by default during installation. To request a CUDA build explicitly:

```bash
CORRDIM_BUILD_CUDA=1 pip install .
```

If PyTorch CUDA and `nvcc` disagree, `setup.py` also supports:

```bash
CORRDIM_FORCE_CUDA_BUILD=1 pip install .
```

Without install-time compilation, the `cuda` backend may still work by JIT-loading sources at runtime or by falling back to PyTorch.

## Build the docs locally

The Read the Docs configuration uses a lightweight docs environment:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser after the build completes.
