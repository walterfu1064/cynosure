"""
Config dataclasses that define the simulation and physical parameters.

The units of dimensional quantities (`object_pixel_size`, `wavelength`, `focal_length`)
must all be identical (and must also be shared by the z-positions), but are arbitrary
within that constraint.
"""

from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """
    Defines meta-aspects of how a model should train/validate.
    Encapsulated here to clean up the ZstackSolver constructors.
    """
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-3
    batch_size: int = 32
    generator_chunk: int = 4  # chunk size for z-stack simulation to bound peak memory use
    steps_per_epoch: int = 200
    val_batches: int = 32
    val_seed: int = 42  # keeps validation passes consistently seeded
    mixing_warmup_epochs: int = 0  # only used in `ZstackSolver_MixedDensity` to prevent premature component collapse


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Parameters relating to the physical situation is digitized"""
    pupil_grid_size: int  # recommended to be `2^n - 1` for some `n`
    object_grid_size: int  # recommended to be `2^n - 1` for some `n`
    object_pixel_size: float
    ftype: torch.dtype = torch.float64
    ctype: torch.dtype = torch.complex128


@dataclass(frozen=True, slots=True)
class OpticalConfig:
    """Parameters related to the physical imaging system"""
    wavelength: float
    focal_length: float
    numerical_aperture: float  # `medium_index * sin(theta_max)`
    aperture_type: Literal["flat", "gaussian", "supergaussian", "fitted"]
    medium_index: float  # emitter's embedding medium
    immersion_index: float | None = None  # objective's immersion (defaults to no interface/no z-rescaling)


@dataclass(frozen=True, slots=True)
class ZernikeConfig:
    """
    Defines a max N for a Zernike bank.
    Optionally only includes specific (n, m) elements.
    """
    max_n: int
    allowed_nm: tuple[tuple[int, int], ...] = ()
    num_elements: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_nm", tuple(tuple(nm) for nm in self.allowed_nm))
        allowed = set(self.allowed_nm)
        count = sum(
            1 for n in range(self.max_n + 1) for m in range(-n, n + 1, 2)
            if not allowed or (n, m) in allowed
        )
        object.__setattr__(self, "num_elements", count)


@dataclass(frozen=True, slots=True)
class PriorConfig:
    """
    Defines statistics for a prior distribution over Zernike coefficients.
    Takes each coefficient to be Gaussian, with RMS decaying with radial
    order `n` as `rms = base_rms / (n+1)^decay_order`.
    """
    base_rms: float
    decay_order: int

    def coef_scales(self, nm_indices: torch.Tensor) -> torch.Tensor:
        """Returns the per-coefficient RMS implied by this prior, given [N, 2] (n, m) indices"""
        n = nm_indices[:, 0]
        return self.base_rms / (n + 1.0) ** self.decay_order


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """
    Noise model to be applied to clean, simulated images.

    To each z-stack will be added:
    - a total signal photons sampled log-uniformly from [min_photons, max_photons]
    - a flat background sampled uniformly from [0, max-background]
    - Poisson shot noise based on the signal + background
    - Gaussian read noise with rms `read_noise`
    """
    min_photons: float
    max_photons: float
    max_background: float
    read_noise: float


@dataclass(frozen=True, slots=True)
class VelocityConfig:
    """
    Defines how a VelocityFlow model should be constructed,
    and how it should be integrated during inference.
    """
    time_embedding_dims: int = 32
    hidden_dims: Sequence[int] = (256, 256, 256)
    is_residual: bool = True
    residual_dims: int = 512
    num_sample_steps: int = 50
