# CorrDim documentation

CorrDim is a Python library for computing the **correlation dimension** of autoregressive language models from next-token log-probability vectors.

This documentation is organized around two goals:

- get you from installation to a first result quickly
- provide a full API reference generated directly from the source tree

## Start here

```{toctree}
:maxdepth: 2
:caption: User guide

installation
concepts
quickstart
cli
examples
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```

## Common entry points

If you want the simplest path from text to a final scalar result, start with `corrdim.measure_text(...)`.

If you want the full correlation-integral curve first, use `corrdim.curve_from_text(...)` or `corrdim.curve_from_vectors(...)`, then fit with `corrdim.estimate_dimension_from_curve(...)`.

If you want to study how structure changes over sequence prefixes, use `corrdim.progressive_curve_from_text(...)` or `corrdim.progressive_correlation_integral(...)`.

If you want **fitted correlation dimensions** at many prefix lengths without separate full runs, use `corrdim.measure_text_progressive(...)` (see the quickstart section on progressive analysis).
