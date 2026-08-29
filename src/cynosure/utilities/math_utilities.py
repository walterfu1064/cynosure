"""
Standalone helper functions for a couple of math ops.
NOT meant for training with, I didn't write these with a eye towards
keeping the gradients free from artifacts (e.g., at 0 for the stdev sqrt).
"""

import torch


def masked_mean(
        data: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
        keepdim: bool = False,
) -> torch.Tensor:
    """
    Calculates the mean along the given dimension,
    ignoring elements where `mask = True`.
    """
    if not data.shape == mask.shape:
        raise ValueError(f"Data shape {data.shape} does not match mask shape {mask.shape}")
    if dim >= 0 and dim >= data.ndim:
        raise ValueError(f"Cannot calculate over {dim} dimension for data of shape {data.shape}")
    if dim < 0 and -dim > data.ndim:
        raise ValueError(f"Cannot calculate over {dim} dimension for data of shape {data.shape}")
    if not data.is_floating_point():
        raise ValueError(f"Data must be floating point, got {data.dtype}")
    if not mask.dtype == torch.bool:
        raise ValueError(f"Mask must be bool, got {mask.dtype}")

    num_valid = (~mask).sum(dim, keepdim=True).to(data.dtype)
    zero = torch.zeros_like(data)

    total = torch.where(mask, zero, data).sum(dim, keepdim=True)
    mean = total / num_valid.clamp(min=1.0)
    if not keepdim:
        mean = mean.squeeze(dim)
    return mean


def masked_stdev(
        data: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
        keepdim: bool = False,
) -> torch.Tensor:
    """
    Calculates the standard deviation along the given dimension,
    ignoring elements where `mask = True`.
    """
    if not data.shape == mask.shape:
        raise ValueError(f"Data shape {data.shape} does not match mask shape {mask.shape}")
    if dim >= 0 and dim >= data.ndim:
        raise ValueError(f"Cannot calculate over {dim} dimension for data of shape {data.shape}")
    if dim < 0 and -dim > data.ndim:
        raise ValueError(f"Cannot calculate over {dim} dimension for data of shape {data.shape}")
    if not data.is_floating_point():
        raise ValueError(f"Data must be floating point, got {data.dtype}")

    num_valid = (~mask).sum(dim, keepdim=True).to(data.dtype)
    zero = torch.zeros_like(data)

    total = torch.where(mask, zero, data).sum(dim, keepdim=True)
    mean = total / num_valid.clamp(min=1.0)

    residual = torch.where(mask, zero, data - mean)
    denominator = num_valid - 1
    variance = residual.square().sum(dim, keepdim=True) / denominator.clamp(min=1.0)
    variance = torch.where(denominator > 0, variance, torch.full_like(variance, torch.nan))
    if not keepdim:
        variance = variance.squeeze(dim)
    return variance.sqrt()
