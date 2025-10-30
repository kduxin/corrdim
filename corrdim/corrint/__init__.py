import torch

CORRINT_BACKEND = "triton"     # "triton" or "pytorch" or "pytorch_fast"

def set_corrint_backend(backend: str = "triton"):
    assert backend in ["triton", "pytorch", "pytorch_fast"], "Invalid backend"
    global CORRINT_BACKEND
    CORRINT_BACKEND = backend

def correlation_integral(vecs: torch.FloatTensor, epsilons: torch.FloatTensor, **kwargs) -> torch.LongTensor:
    if CORRINT_BACKEND == "triton":
        from ._corrint_triton import correlation_integral as corrint
        return corrint(vecs, epsilons, **kwargs)
    elif CORRINT_BACKEND in ["pytorch", "pytorch_fast"]:
        from ._corrint_pytorch import correlation_integral as corrint_pytorch
        if CORRINT_BACKEND == "pytorch_fast":
            return corrint_pytorch(vecs, epsilons, fast=True, **kwargs)
        else:
            return corrint_pytorch(vecs, epsilons, **kwargs)
    else:
        raise ValueError(f"Invalid backend: {CORRINT_BACKEND}")
