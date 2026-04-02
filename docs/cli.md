# Command-line interface

CorrDim ships with a `corrdim` CLI entry point.

## Available commands

- `measure-text`: read a text file and return the fitted correlation dimension
- `curve-from-text`: read a text file and return the full curve as JSON
- `estimate-dimension`: load a saved curve JSON file and fit the dimension

## Measure a text file

```bash
corrdim measure-text \
  --file sample.txt \
  --model Qwen/Qwen2.5-1.5B
```

Write the JSON result to a file:

```bash
corrdim measure-text \
  --file sample.txt \
  --model Qwen/Qwen2.5-1.5B \
  --output result.json
```

## Export the full curve

```bash
corrdim curve-from-text \
  --file sample.txt \
  --model Qwen/Qwen2.5-1.5B \
  --output curve.json
```

## Fit a dimension from saved curve data

```bash
corrdim estimate-dimension \
  --curve-json curve.json \
  --output fit.json
```

## Common options

The text-oriented commands share these useful flags:

- `--context-length`
- `--stride`
- `--dim-reduction`
- `--dtype`
- `--num-epsilon`
- `--backend`
- `--line-sep`

For `--dim-reduction`, values like `8192` are accepted, and `none` disables reduction.
