"""
Contains some reasonable default configs for setting up ZstackSolver models.
Used as convenience methods for some of the examples, but there's nothing particularly deep or special about these.

I could set these as defaults in the actual config dataclasses, but that would make them too
specific to the examples I arbitrarily chose. Gross. This way is kind of annoying, but I like
it better than the alternative.
"""

from typing import Literal, Optional, Sequence

import torch

from cynosure.config import (
    NoiseConfig,
    OpticalConfig,
    PriorConfig,
    SimulationConfig,
    TrainingConfig,
    VelocityConfig,
    ZernikeConfig,
)


def get_default_training_config(
        learning_rate: float = 1.0e-3,
        weight_decay: float = 1.0e-3,
        batch_size: int = 32,
        generator_chunk: int = 4,
        steps_per_epoch: int = 200,
        val_batches: int = 32,
        val_seed: int = 42,
        mixing_warmup_epochs: int = 10,
        min_allocation: float = 0.05,
        mixing_entropy_weight: float = 0.5,
) -> TrainingConfig:
    return TrainingConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        generator_chunk=generator_chunk,
        steps_per_epoch=steps_per_epoch,
        val_batches=val_batches,
        val_seed=val_seed,
        mixing_warmup_epochs=mixing_warmup_epochs,
        min_allocation=min_allocation,
        mixing_entropy_weight=mixing_entropy_weight,
    )


def get_default_simulation_config(
        pupil_grid_size: int = 255,
        object_grid_size: int = 31,
        object_pixel_size: float = 0.15,
        ftype: torch.dtype = torch.float32,
        ctype: torch.dtype = torch.complex64,
) -> SimulationConfig:
    return SimulationConfig(
        pupil_grid_size=pupil_grid_size,
        object_grid_size=object_grid_size,
        object_pixel_size=object_pixel_size,
        ftype=ftype,
        ctype=ctype,
    )


def get_default_optical_config(
        wavelength: float = 0.510,
        focal_length: float = 4.50e3,
        numerical_aperture: float = 0.9,
        aperture_type: Literal["flat", "gaussian", "supergaussian", "fitted"] = "fitted",
        medium_index: float = 1.54,
        immersion_index: Optional[float] = 1.0
) -> OpticalConfig:
    return OpticalConfig(
        wavelength=wavelength,
        focal_length=focal_length,
        numerical_aperture=numerical_aperture,
        aperture_type=aperture_type,
        medium_index=medium_index,
        immersion_index=immersion_index,
    )


def get_default_phase_config(
        max_n: int = 6,
        allowed_nm: tuple[tuple[int, int], ...] = (
                (1, 1), (1, -1),
                (2, -2), (2, 0), (2, 2),
                (3, -3), (3, -1), (3, 1), (3, 3),
                (4, -4), (4, -2), (4, 0), (4, 2), (4, 4),
                (5, -5), (5, -3), (5, -1), (5, 1), (5, 3), (5, 5),
        ),
) -> ZernikeConfig:
    return ZernikeConfig(max_n=max_n, allowed_nm=allowed_nm)


def get_default_amplitude_config(
        max_n: int = 4,
        allowed_nm: tuple[tuple[int, int], ...] = ((0, 0), (2, 0), (4, 0)),
) -> ZernikeConfig:
    return ZernikeConfig(max_n=max_n, allowed_nm=allowed_nm)


def get_default_phase_prior(
        base_rms: float = 0.05,
        decay_order: int = 1,
) -> PriorConfig:
    return PriorConfig(base_rms=base_rms, decay_order=decay_order)


def get_default_amplitude_prior(
    base_rms: float = 0.05,
    decay_order: int = 2,
) -> PriorConfig:
    return PriorConfig(base_rms=base_rms, decay_order=decay_order)


def get_default_noise_config(
        min_photons: int = 100_000,
        max_photons: int = 1_000_000,
        max_background: int = 1_000,
        read_noise: int = 10,
) -> NoiseConfig:
    return NoiseConfig(
        min_photons=min_photons,
        max_photons=max_photons,
        max_background=max_background,
        read_noise=read_noise,
    )


def get_default_velocity_config(
        time_embedding_dims: int = 64,
        hidden_dims: Sequence[int] = (256, 256, 256),
        is_residual: bool = True,
        residual_dims: int = 256,
        num_sample_steps: int = 50,
) -> VelocityConfig:
    return VelocityConfig(
        time_embedding_dims=time_embedding_dims,
        hidden_dims=hidden_dims,
        is_residual=is_residual,
        residual_dims=residual_dims,
        num_sample_steps=num_sample_steps,
    )

