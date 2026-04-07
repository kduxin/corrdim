# Examples

The repository already includes runnable examples in `examples/basic_usage.py`.

That example script is organized into three workflows:

- high-level API: text directly to correlation dimension
- low-level API: model logits to vectors to curve to fitted dimension
- progressive analysis: correlation dimension as the sequence grows

## Running the example script

From the repository root:

```bash
python examples/basic_usage.py
```

The script saves plots under `examples/plots/` when the required runtime dependencies and model weights are available.

## Good starting points

If you are new to the project:

- start with `measure_text(...)` for an end-to-end workflow
- switch to `curve_from_text(...)` when you want to inspect the curve itself
- use `progressive_curve_from_text(...)` when you care about changes over sequence prefixes
- use `measure_text_progressive(...)` when you want multiple `DimensionResult` values (fitted dimensions) along prefixes in one model pass

## API reference

For a full symbol-by-symbol reference, see the generated API section in the sidebar.
