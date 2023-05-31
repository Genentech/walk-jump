import torch


def isotropic_gaussian_noise_like(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return sigma * torch.randn_like(x.float())
