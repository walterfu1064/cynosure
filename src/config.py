from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SimulationConfig:
    """Parameters relating to the physical situation is digitized"""
    pupil_grid_size: int  # recommended to be `2^n - 1` for some `n`
    object_grid_size: int  # recommended to be `2^n - 1` for some `n`
    object_pixel_size: float


@dataclass
class OpticalConfig:
    """Parameters related to the physical imaging system"""
    wavelength: float
    focal_length: float
    numerical_aperture: float
    aperture_type: Literal["flat", "gaussian", "supergaussian", "fitted"]


@dataclass
class ZernikeConfig:
    """
    Defines a max N for a Zernike bank.
    Optionally only includes specific (n, m) elements.
    """
    max_n: int
    allowed_nm: list[tuple[int, int]] = field(default_factory=list)
