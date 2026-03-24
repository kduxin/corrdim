from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
import fcntl

import torch
from torch.utils.cpp_extension import load

_EXT_NAME = "corrdim_corrint_cuda_ext"
_EXT_MOD: Optional[object] = None
_EXT_LOAD_ERROR: Optional[Exception] = None


@contextmanager
def _jit_build_lock():
    base = os.environ.get("TORCH_EXTENSIONS_DIR")
    if base is None or not base.strip():
        base = str(Path.home() / ".cache" / "torch_extensions")
    lock_dir = Path(base)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{_EXT_NAME}.lock"
    with open(lock_path, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_extension() -> object:
    global _EXT_MOD, _EXT_LOAD_ERROR
    if _EXT_MOD is not None:
        return _EXT_MOD
    if _EXT_LOAD_ERROR is not None:
        raise RuntimeError(f"CUDA extension load failed: {_EXT_LOAD_ERROR}") from _EXT_LOAD_ERROR

    # Prefer install-time compiled extension (built during `uv sync`).
    try:
        _EXT_MOD = importlib.import_module("corrdim.corrint._cuda_ext")
        return _EXT_MOD
    except Exception:
        pass

    base = Path(__file__).resolve().parent / "ext"
    cpp = str(base / "corrdim_cuda.cpp")
    cu = str(base / "corrdim_cuda_kernel.cu")

    try:
        # Prevent multi-process races building the same torch extension.
        with _jit_build_lock():
            _EXT_MOD = load(
                name=_EXT_NAME,
                sources=[cpp, cu],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
        return _EXT_MOD
    except Exception as exc:  # pragma: no cover
        _EXT_LOAD_ERROR = exc
        raise

