from __future__ import annotations

import os
import subprocess
from pathlib import Path

from setuptools import setup


def _parse_nvcc_version(cuda_home: str | None) -> str | None:
    if not cuda_home:
        return None
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    if not nvcc.exists():
        return None
    try:
        out = subprocess.check_output([str(nvcc), "--version"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None
    for line in out.splitlines():
        if "release " in line:
            # Example: "Cuda compilation tools, release 12.1, V12.1.66"
            token = line.split("release ", 1)[1].split(",", 1)[0].strip()
            return token
    return None


def _build_cuda_extensions():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    import torch

    root = Path(__file__).resolve().parent
    sources = [
        str(root / "corrdim" / "corrint" / "ext" / "corrdim_cuda.cpp"),
        str(root / "corrdim" / "corrint" / "ext" / "corrdim_cuda_kernel.cu"),
    ]

    force = os.environ.get("CORRDIM_FORCE_CUDA_BUILD", "").strip() == "1"
    if CUDA_HOME is None and not force:
        return [], {}

    torch_cuda = (torch.version.cuda or "").strip()
    nvcc_cuda = (_parse_nvcc_version(CUDA_HOME) or "").strip()
    if (not force) and torch_cuda and nvcc_cuda:
        # Compare major.minor only.
        torch_mm = ".".join(torch_cuda.split(".")[:2])
        nvcc_mm = ".".join(nvcc_cuda.split(".")[:2])
        if torch_mm != nvcc_mm:
            print(
                f"[corrdim] Skip CUDA extension build: nvcc={nvcc_cuda} "
                f"!= torch CUDA={torch_cuda}. Set CORRDIM_FORCE_CUDA_BUILD=1 to force."
            )
            return [], {}

    ext_modules = [
        CUDAExtension(
            name="corrdim.corrint._cuda_ext",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
    return ext_modules, cmdclass


ext_modules, cmdclass = _build_cuda_extensions()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

