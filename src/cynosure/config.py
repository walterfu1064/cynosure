"""
Config dataclasses that define the simulation and physical parameters.

The units of dimensional quantities (`object_pixel_size`, `wavelength`, `focal_length`)
must all be identical (and must also be shared by the z-positions), but are arbitrary
within that constraint.
"""

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import torch
import yaml


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """
    Defines meta-aspects of how a model should train/validate.
    Encapsulated here to clean up the ZstackSolver constructors.
    """
    learning_rate: float
    weight_decay: float
    batch_size: int
    generator_chunk: int  # chunk size for z-stack simulation to bound peak memory use
    steps_per_epoch: int
    val_batches: int
    val_seed: int  # keeps validation passes consistently seeded


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Parameters relating to the physical situation is digitized"""
    pupil_grid_size: int  # recommended to be `2^n - 1` for some `n`
    object_grid_size: int  # recommended to be `2^n - 1` for some `n`
    object_pixel_size: float
    ftype: torch.dtype
    ctype: torch.dtype


@dataclass(frozen=True, slots=True)
class OpticalConfig:
    """Parameters related to the physical imaging system"""
    wavelength: float
    focal_length: float
    numerical_aperture: float  # `medium_index * sin(theta_max)`
    aperture_type: Literal["flat", "gaussian", "supergaussian", "fitted"]
    medium_index: float  # emitter's embedding medium
    immersion_index: Optional[float]  # objective's immersion (defaults to no interface/no z-rescaling)


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
class BlobPriorConfig:
    """
    Defines statistics for a prior distribution over `SampledKBlobs` parameters.

    Each blob is parameterized by its center row, center column, FWHM diameter, and amplitude.
    Positions are represented by their offsets from the grid center, diameter by its
    natural log, and amplitude by its pre-sigmoid logit. Each such representation is taken
    to be Gaussian about zero, with an RMS as defined here.
    """
    position_sigma: float  # physical units
    reference_diameter: float  # physical units
    log_diameter_sigma: float
    amplitude_logit_sigma: float

    def param_scales(self, num_blobs: int) -> torch.Tensor:
        """
        Returns the per-parameter prior RMS, in label order [row, col, log-diam, amp-logit].

        Blobs are canonically ordered by descending amplitude, with the brightest pinned to
        exactly 1 and its logit dropped from the labels (since absolute brightness will get
        pinned by the unit-sum normalization).
        """
        return torch.cat([
            torch.full((num_blobs,), self.position_sigma),
            torch.full((num_blobs,), self.position_sigma),
            torch.full((num_blobs,), self.log_diameter_sigma),
            torch.full((num_blobs - 1,), self.amplitude_logit_sigma),
        ])


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """
    Noise model to be applied to clean, simulated images.

    To each z-stack will be added:
    - a per-frame signal photon count sampled log-uniformly from [min_photons, max_photons]
    - a flat background sampled uniformly from [0, max-background]
    - Poisson shot noise based on the signal + background
    - Gaussian read noise with rms `read_noise`
    """
    min_photons: float
    max_photons: float
    max_background: float
    read_noise: float

    @property
    def average_photons(self) -> float:
        """Log-mean number of photons per frame, for a deterministic scale"""
        return math.sqrt(self.max_photons * self.min_photons)

    @property
    def average_background(self) -> float:
        """Mean background, for a deterministic scale"""
        return self.max_background / 2


@dataclass(frozen=True, slots=True)
class MixtureConfig:
    """
    Defines how a Mixture-Density Network should be constructed and trained.
    """
    num_components: int
    mixing_warmup_epochs: int  # weights held uniform this long, to prevent premature component collapse
    min_allocation: float  # floor on each component's responsibility, to avoid killing one entirely
    mixing_entropy_weight: float  # mixing-weight entropy bonus (opposes component collapse)


@dataclass(frozen=True, slots=True)
class VelocityConfig:
    """
    Defines how a VelocityFlow model should be constructed, and how it should be integrated during inference.
    """
    time_embedding_dims: int
    hidden_dims: Sequence[int]
    is_residual: bool
    residual_dims: int
    num_sample_steps: int


# Allow dtypes to get saved by Lightning as hyperparams
yaml.add_representer(torch.dtype, lambda dumper, dtype: dumper.represent_str(str(dtype)))

# These dataclasses are frozen and hold numbers, mark them save to serialize so `torch.load` doesn't complain
torch.serialization.add_safe_globals([
    MixtureConfig,
    NoiseConfig,
    OpticalConfig,
    PriorConfig,
    SimulationConfig,
    TrainingConfig,
    VelocityConfig,
    ZernikeConfig,
])
