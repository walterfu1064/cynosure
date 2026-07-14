"""
Camera/photon noise model for corrupting clean, simulated image stacks.

To each z-stack will be added:
- a total signal photons sampled log-uniformly from [min_photons, max_photons]
- a flat background sampled uniformly from [0, max-background]
- Poisson shot noise based on the signal + background
- Gaussian read noise with rms `read_noise`
"""

import math
from typing import Optional

import torch

from ..config import NoiseConfig


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
        Adds noise to a stack or batch of stacks, [..., num_z, H, W].
        If batched, photon count and background are sampled independently for each stack.
        """
        sample_shape = (*images.shape[:-3], 1, 1, 1)
        rand_kwargs = {"generator": generator, "dtype": images.dtype, "device": images.device}

        log_range = self.log_max_photons - self.log_min_photons
        photons = torch.exp(self.log_min_photons + log_range * torch.rand(sample_shape, **rand_kwargs))
        background = self.max_background * torch.rand(sample_shape, **rand_kwargs)

        signal = images.clamp(min=0)  # clip any small negatives from the FFT convolution
        signal = signal * photons / signal.sum((-3, -2, -1), keepdim=True)
        noisy = torch.poisson(signal + background, generator=generator)
        if self.read_noise > 0:
            noisy = noisy + self.read_noise * torch.randn(images.shape, **rand_kwargs)
        return noisy
