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
    allowed_nm: list[tuple[int, int]] = field(default_factory=list)
