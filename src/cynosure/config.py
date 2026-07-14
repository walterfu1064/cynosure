"""
Config dataclasses that define the simulation and physical parameters.

The units of dimensional quantities (`object_pixel_size`, `wavelength`, `focal_length`)
must all be identical (and must also be shared by the z-positions), but are arbitrary
within that constraint.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch


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
