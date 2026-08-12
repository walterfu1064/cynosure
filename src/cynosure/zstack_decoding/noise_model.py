"""
Camera/photon noise model for corrupting clean, simulated image stacks.

To each z-stack will be added:
- a signal photon count sampled log-uniformly from [min_photons, max_photons]
- a flat background sampled uniformly from [0, max-background]
- Poisson shot noise based on the signal + background
- Gaussian read noise with rms `read_noise`

Photon count and background will be denoted per-frame, and will be constant over a z-stack.
"""

import math
from typing import Optional

import torch

from ..config import NoiseConfig


def _as_per_zstack(value: torch.Tensor | float, images: torch.Tensor) -> torch.Tensor:
    """
    Reshapes a photon budget or background level so it broadcasts across `images`.

    `images` should be a [..., Z, H, W] tensor representing a z-stack, a batch of z-stacks, etc.
    The given value is shaped to a [..., 1, 1, 1] tensor so it can represent a set of per z-stack values.
    """
    value = torch.as_tensor(value, dtype=images.dtype, device=images.device)
    batch_shape = images.shape[:-3]
    if value.shape not in (torch.Size(()), batch_shape):
        raise ValueError(f"Unable to reshape {value.shape} to match {images.shape = } per-z-stack")
    return value.reshape(*value.shape, 1, 1, 1)


def scale_to_photon_counts(
        images: torch.Tensor,
        photons: torch.Tensor | float,
        background: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """
    Scales clean images to an expected number of photons.

    This is intended to be run on z-stacks, where the trailing dims of `images` are [Z, H, W].
    Single images or batches of single images should be unsqueezed to [..., 1, H, W] before
    passing to this function.

    Arguments:
    - images: [..., Z, H, W]
    - photons: scalar, or [...,] matching the batch dims of `images`
    - background: scalar, or [...,] matching the batch dims of `images`
    Returns:
    - same as input `images`
    """
    if images.ndim < 3:
        raise ValueError(f"`images` must be at least [Z, H, W], got {images.shape}")
    signal = images.clamp(min=0)  # clip any small negatives from the FFT convolution
    signal = signal * _as_per_zstack(photons, images) / signal.sum((-2, -1), keepdim=True)
    return signal + _as_per_zstack(background, images)


class NoiseModel(torch.nn.Module):
    def __init__(self, noise_cfg: NoiseConfig):
        super().__init__()

        if not 0 < noise_cfg.min_photons <= noise_cfg.max_photons:
            raise ValueError(
                f"Noise bounds must satisfy 0 < min_photons <= max_photons, "
                f"got {noise_cfg.min_photons = } and {noise_cfg.max_photons = }"
            )
        if noise_cfg.max_background < 0:
            raise ValueError(f"max_background must be non-negative (got {noise_cfg.max_background})")
        if noise_cfg.read_noise < 0:
            raise ValueError(f"read_noise must be non-negative (got {noise_cfg.read_noise})")

        self.noise_cfg = noise_cfg
        self.log_min_photons = math.log(noise_cfg.min_photons)
        self.log_max_photons = math.log(noise_cfg.max_photons)
        self.max_background = noise_cfg.max_background
        self.read_noise = noise_cfg.read_noise

    def forward(
            self,
            images: torch.Tensor,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Adds noise to a stack or batch of stacks, [..., Z, H, W].
        If batched, photon count and background are sampled independently for each stack.
        """
        batch_shape = images.shape[:-3]
        rand_kwargs = {"generator": generator, "dtype": images.dtype, "device": images.device}

        log_range = self.log_max_photons - self.log_min_photons
        photons = torch.exp(self.log_min_photons + log_range * torch.rand(batch_shape, **rand_kwargs))
        background = self.max_background * torch.rand(batch_shape, **rand_kwargs)

        noisy = torch.poisson(scale_to_photon_counts(images, photons, background), generator=generator)
        if self.read_noise > 0:
            noisy = noisy + self.read_noise * torch.randn(images.shape, **rand_kwargs)
        return noisy
