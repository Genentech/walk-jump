from typing import Callable

import torch

from walkjump.model import TrainableScoreModel


def walk(
    seed_tensor: torch.Tensor,
    model: TrainableScoreModel,
    sampler_fn: Callable,
    chunksize: int = 1,
    save_trajectory: bool = False,
) -> torch.Tensor:
    """
    Walk step

    Parameters
    ----------
    seed_tensor: torch.Tensor
        Stacked seed batch
    model: ScoreModel
        Model of Y-manifold used for walking
    sampler_fn: Callable
        The (partial) function with sampling parameters
    chunksize: int
        Used for chunking the batch to save memory. Providing
        chunksize = N will force the sampling to occur in N batches.

    Returns
    -------
    torch.Tensor
        Samples from Y
    """
    seed_tensor = seed_tensor.to(model.device)
    list_ys = []

    for seed_chunk in seed_tensor.chunk(chunksize):

        # note: apply_noise should control whether seed_chunk.requires_grad
        seed_chunk = model.apply_noise(seed_chunk)
        # seed_chunk.requires_grad = True
        ys, *v_trajectory = sampler_fn(
            model, seed_chunk, torch.zeros_like(seed_chunk), save_trajectory=save_trajectory
        )
        if save_trajectory:
            list_ys.extend(v_trajectory[-1])
        else:
            list_ys.append(ys.detach())

    ys = torch.cat(list_ys, dim=0)

    return ys
