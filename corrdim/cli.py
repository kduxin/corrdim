"""Command-line interface for corrdim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .dimension import estimate_dimension_from_curve
from .high_level import measure_text
from .low_level import curve_from_text
from .types import CurveResult


def _parse_dtype(dtype: str):
    if dtype is None:
        return None
    s = str(dtype).strip().lower()
    mapping = {
        "float16": "float16",
        "fp16": "float16",
        "half": "float16",
        "float32": "float32",
        "fp32": "float32",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
    }
    s = mapping.get(s, s)
    try:
        return getattr(__import__("torch"), s)
    except Exception as e:
        raise ValueError(f"Unsupported dtype={dtype!r}. Use one of: float16/float32/bfloat16.") from e


def _parse_stride(stride: str):
    if stride is None:
        return 1
    s = str(stride).strip()
    try:
        v = int(s)
    except ValueError as e:
        raise ValueError("stride must be a positive integer") from e
    if v < 1:
        raise ValueError("stride must be a positive integer")
    return v


def _parse_dim_reduction(dim_reduction):
    if dim_reduction is None:
        return 8192
    s = str(dim_reduction).strip().lower()
    if s in {"none", "null", "off"}:
        return None
    try:
        v = int(s)
    except ValueError as e:
        raise ValueError("dim-reduction must be an integer or 'none'") from e
    if v < 1:
        raise ValueError("dim-reduction must be a positive integer or 'none'")
    return v


def _read_text_file(path: str, line_sep: str) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except Exception as exc:
        raise ValueError(f"Error reading file {path}: {exc}") from exc
    if not text:
        raise ValueError(f"File {path} is empty")
    return line_sep.join(text.split("\n"))


def _add_common_text_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--file", required=True, help="Text file to analyze")
    parser.add_argument("--model", "-m", required=True, help="Model name for from_pretrained")
    parser.add_argument("--line-sep", default="\n")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dim-reduction", type=str, default="8192")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num-epsilon", type=int, default=1024)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--output", "-o", default=None)


def _write_or_print(payload: str, output: str | None) -> None:
    if output:
        Path(output).write_text(payload, encoding="utf-8")
        print(f"Results saved to {output}")
    else:
        print(payload)


def _cmd_measure_text(args: argparse.Namespace) -> None:
    text = _read_text_file(args.file, args.line_sep)
    result = measure_text(
        text=text,
        model=args.model,
        context_length=args.context_length,
        stride=_parse_stride(args.stride),
        dim_reduction=_parse_dim_reduction(args.dim_reduction),
        precision=_parse_dtype(args.dtype),
        num_epsilon=args.num_epsilon,
        backend=args.backend,
    )
    payload = {
        "sequence_length": result.sequence_length,
        "corrdim": result.corrdim,
        "fit_r2": result.fit_r2,
        "linear_region_bounds": result.linear_region_bounds,
    }
    _write_or_print(json.dumps(payload, ensure_ascii=False, indent=2), args.output)


def _cmd_curve_from_text(args: argparse.Namespace) -> None:
    text = _read_text_file(args.file, args.line_sep)
    curve = curve_from_text(
        text=text,
        model=args.model,
        context_length=args.context_length,
        stride=_parse_stride(args.stride),
        dim_reduction=_parse_dim_reduction(args.dim_reduction),
        precision=_parse_dtype(args.dtype),
        num_epsilon=args.num_epsilon,
        backend=args.backend,
    )
    payload = {
        "sequence_length": curve.sequence_length,
        "epsilons": curve.epsilons.tolist(),
        "corrints": curve.corrints.tolist(),
    }
    _write_or_print(json.dumps(payload, ensure_ascii=False), args.output)


def _cmd_estimate_dimension(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.curve_json).read_text(encoding="utf-8"))
    curve = CurveResult(
        sequence_length=int(data["sequence_length"]),
        epsilons=np.asarray(data["epsilons"], dtype=float),
        corrints=np.asarray(data["corrints"], dtype=float),
    )
    result = estimate_dimension_from_curve(curve, correlation_integral_range="auto")
    payload = {
        "sequence_length": result.sequence_length,
        "corrdim": result.corrdim,
        "fit_r2": result.fit_r2,
        "linear_region_bounds": result.linear_region_bounds,
    }
    _write_or_print(json.dumps(payload, ensure_ascii=False, indent=2), args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="CorrDim CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure_text_parser = subparsers.add_parser("measure-text", help="Measure correlation dimension from text")
    _add_common_text_args(measure_text_parser)
    measure_text_parser.set_defaults(handler=_cmd_measure_text)

    curve_text_parser = subparsers.add_parser("curve-from-text", help="Compute correlation-integral curve from text")
    _add_common_text_args(curve_text_parser)
    curve_text_parser.set_defaults(handler=_cmd_curve_from_text)

    estimate_dim_parser = subparsers.add_parser("estimate-dimension", help="Estimate dimension from curve JSON")
    estimate_dim_parser.add_argument("--curve-json", required=True, help="Curve JSON file")
    estimate_dim_parser.add_argument("--output", "-o", default=None)
    estimate_dim_parser.set_defaults(handler=_cmd_estimate_dimension)

    args = parser.parse_args()
    try:
        args.handler(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
