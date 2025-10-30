import tqdm
import numpy as np
import torch

def correlation_integral(vecs: torch.FloatTensor, epsilons: torch.FloatTensor, show_progress: bool = False, block_size: int = 512, fast: bool = False) -> torch.LongTensor:
    N = vecs.shape[0]

    epsilons_log10 = epsilons.log10().cpu().numpy()

    counts = np.zeros(epsilons.shape[0], dtype=np.int64)
    for i in tqdm.trange(0, N, block_size, disable=not show_progress, desc="Computing correlation integrals"):
        slici = vecs[i : i + block_size].to(torch.float32)

        for j in range(0, N, block_size):
            slicj = vecs[j : j + block_size].to(torch.float32)

            if fast:
                distances = torch.cdist(slici, slicj, p=2, compute_mode="use_mm_for_euclid_dist_if_necessary")
            else:
                distances = torch.cdist(slici, slicj, p=2, compute_mode="donot_use_mm_for_euclid_dist")
            # (block_size, block_size)

            assert not (
                torch.isnan(distances).any() or torch.isinf(distances).any()
            ), "Found nan or inf in distances. Please consider raising the precision for distance computation."

            if i == j:
                distances[torch.arange(len(slici)), torch.arange(len(slici))] = float("Inf")

            distances_log10 = (
                torch.sort(distances.view(-1), descending=False, stable=False).values.data.log10().cpu().numpy()
            )

            counts_batch = np.searchsorted(distances_log10, epsilons_log10)
            counts += counts_batch

    counts = torch.tensor(counts, dtype=torch.float64, device=vecs.device)
    corrints = counts / (N * (N - 1) / 2)
    return corrints