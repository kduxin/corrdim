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

## Build the docs locally

The Read the Docs configuration uses a lightweight docs environment:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser after the build completes.
