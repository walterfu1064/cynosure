"""
Models of an object to be imaged.

Each subclass implements `sample()`, which returns a [B, 1, H, W]
batch of object distributions to be convolved with the (possibly multi-z) PSF.

This sampling and convolution can be done in a single call using `forwards()`.

Subclasses will implement different static or learnable object shapes.
"""

from abc import ABC, abstractmethod
import math
from typing import Optional

import torch
import torch.nn as nn

from ..config import SimulationConfig
from ..utilities.fft_utilities import next_power_of_2


def make_radial_field(
        grid_size: int,
        pixel_size: float,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Returns a [grid_size, grid_size] tensor giving distance from the grid center in physical units"""
    coords = (torch.arange(grid_size, dtype=dtype) - grid_size // 2) * pixel_size
    Y, X = torch.meshgrid(coords, coords, indexing="ij")
    R = torch.sqrt(X * X + Y * Y)
    return R.to(dtype)


class ObjectDistribution(nn.Module, ABC):
    """
    Base class for object distributions.
    Handles bookkeeping for the object grid and the PSF-convolving FFT grid.
    Subclasses implement `sample`, which returns a batch of object distributions (either identical
    across the batch for static distributions, or independently resampled for stochastic ones).
    `forward` composes that draw with the PSF convolution.
    """

    def __init__(self, sim_cfg: SimulationConfig):
        super().__init__()
        self.sim_cfg = sim_cfg
        self.fft_size = self._calculate_fft_size()

    def _calculate_fft_size(self) -> int:
        """Precalculate FFT size for convolution, including padding to avoid circular artifacts"""
        return next_power_of_2(2 * self.size - 1)

    @property
    def size(self) -> int:
        """Convenience accessor for the side length of the object grid"""
        return self.sim_cfg.object_grid_size

    @property
    def shape(self) -> tuple[int, int]:
        """Convenience accessor for the shape of the object grid"""
        return self.size, self.size

    @property
    def fft_shape(self) -> tuple[int, int]:
        return self.fft_size, self.fft_size

    @abstractmethod
    def sample(
            self,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Returns a [B, 1, H, W] batch of object distributions, broadcastable against a batch of z-stacks.
        Singlet dimensions should be used to keep this shape.
        Identical across batch for static distributions, independently sampled for stochastic ones.
        """
        ...

    def forward(
            self,
            psf: torch.Tensor,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Draws a batch of object distributions and convolves it with `psf`.

        Arguments:
        - psf: [B, Z, H, W] from a BeamPropagator
        - batch_size: number of z-stacks to draw objects for, must match the leading dimension of `psf`
        Returns:
        - [B, Z, H, W] images
        """
        return self.convolve_psf(psf, self.sample(batch_size, generator))

    @staticmethod
    def _repeat_static(distribution: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Broadcasts a static [H, W] distribution to [B, 1, H, W] and returns as a view"""
        return distribution.expand(batch_size, 1, -1, -1)

    def convolve_psf(self, psf: torch.Tensor, distribution: torch.Tensor) -> torch.Tensor:
        """
        Convolves a batch of object distributions with a batch of PSFs.

        Arguments:
        - psf: [B, Z, H, W] with its origin at (H//2, W//2) (the natural viewing order, but not the natural FFT order)
        - distribution: [B, 1, H, W] as returned by `sample`, broadcastable against `psf`
        Returns:
        - [B, Z, H, W] convolved objects
        """

        if psf.ndim != 4:
            raise ValueError(f"PSF must be [B, Z, H, W], got {psf.shape}")
        if distribution.ndim != 4:
            raise ValueError(f"Object distribution must be [B, Z, H, W], got {distribution.shape}")

        if psf.shape[-2:] != self.shape:
            raise ValueError(f"PSF spatial dims {psf.shape[-2:]} does not match object shape {self.shape}")
        if distribution.shape[-2:] != self.shape:  # should never happen if `convolve_psf` is only called internally
            raise ValueError(f"Object spatial dims {psf.shape[-2:]} does not match object shape {self.shape}")

        try:
            torch.broadcast_shapes(distribution.shape[:2], psf.shape[:2])
        except RuntimeError as error:
            raise ValueError(f"Cannot broadcast {distribution.shape} object against {psf.shape} PSF") from error

        psf_padded = psf.new_zeros(*psf.shape[:-2], *self.fft_shape)
        psf_padded[..., :self.size, :self.size] = psf
        psf_padded = torch.roll(psf_padded, shifts=(-(self.size // 2), -(self.size // 2)), dims=(-2, -1))
        psf_fft = torch.fft.rfft2(psf_padded, dim=(-2, -1))

        obj_fft = torch.fft.rfft2(distribution, s=self.fft_shape, dim=(-2, -1))
        result = torch.fft.irfft2(obj_fft * psf_fft, s=self.fft_shape, dim=(-2, -1))
        return result[..., :self.size, :self.size]


class FixedBead(ObjectDistribution):
    """A uniform, round bead with a known diameter"""
    def __init__(
            self,
            sim_cfg: SimulationConfig,
            bead_diameter: float,
    ):
        super().__init__(sim_cfg)
        if bead_diameter <= 0:
            raise ValueError(f"Bead diameter bound must be positive, got {bead_diameter}")
        self.bead_diameter = bead_diameter
        self.register_buffer("_object_distribution", self._make_bead())

    def _make_bead(self) -> torch.Tensor:
        R = make_radial_field(self.size, self.sim_cfg.object_pixel_size, self.sim_cfg.ftype)
        bead = (R <= self.bead_diameter/2)
        bead = bead.to(self.sim_cfg.ftype)
        return bead

    def sample(
            self,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Returns the stored distribution, repeated to [B, 1, H, W].
        Deterministic, so `generator` is unused.
        """
        return self._repeat_static(self._object_distribution, batch_size)


class ParametricBead(ObjectDistribution):
    """A uniform, circular bead of fitted diameter (defined as FWHM)"""
    def __init__(
            self,
            sim_cfg: SimulationConfig,
            bead_diameter_bounds: tuple[float, float],
    ):
        super().__init__(sim_cfg)
        if any(bound <= 0 for bound in bead_diameter_bounds):
            raise ValueError(f"Bead diameter bounds must be positive, got {bead_diameter_bounds}")
        if bead_diameter_bounds[0] > bead_diameter_bounds[1]:
            raise ValueError(f"Bead diameter min must be <= max, got {bead_diameter_bounds}")
        self.bead_diameter_min, self.bead_diameter_max = bead_diameter_bounds
        self.bead_logit = nn.Parameter(torch.zeros((), dtype=self.sim_cfg.ftype))
        self.register_buffer("R", make_radial_field(
            self.size, self.sim_cfg.object_pixel_size, self.sim_cfg.ftype
        ))

    @property
    def bead_diameter(self) -> torch.Tensor:
        """Converts the bead diameter logit to the actual diameter, bounded by the object's min/max"""
        return (
                (self.bead_diameter_max - self.bead_diameter_min) * torch.sigmoid(self.bead_logit)
                + self.bead_diameter_min
        )

    def _make_bead(self) -> torch.Tensor:
        bead = torch.exp(-math.log(2) * torch.pow(self.R / (self.bead_diameter / 2), 4))
        bead = bead.to(self.sim_cfg.ftype)
        return bead

    def sample(
            self,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Returns the rasterized bead, repeated to [B, 1, H, W].
        Deterministic, so `generator` is unused.
        """
        bead = self._make_bead()
        return self._repeat_static(bead, batch_size)


class FixedObject(ObjectDistribution):
    """A known, but arbitrary, object distribution, supplied directly as a [H, W] tensor"""
    def __init__(
            self,
            sim_cfg: SimulationConfig,
            object_distribution: torch.Tensor,
    ):
        super().__init__(sim_cfg)
        if object_distribution.ndim != 2:
            raise ValueError(f"Object distribution must be [H, W], got {tuple(object_distribution.shape)}")
        if tuple(object_distribution.shape) != self.shape:
            raise ValueError(
                f"Object distribution shape {tuple(object_distribution.shape)} "
                f"does not match the object grid {self.shape}"
            )

        self.register_buffer("_object_distribution", object_distribution.to(sim_cfg.ftype))

    def sample(
            self,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Returns the stored distribution, repeated to [B, 1, H, W].
        Deterministic, so `generator` is unused.
        """
        return self._repeat_static(self._object_distribution, batch_size)


class FreeField(ObjectDistribution):
    """
    An unconstrained, fittable object distribution.

    Represents the object as a tensor of logits on the object grid.
    Sigmoiding and then normalizing to unity sum (to match the pinned amplitude piston)
    yields the object distribution itself, for convolving with the PSF.
    """
    def __init__(
            self,
            sim_cfg: SimulationConfig,
            initial_speckle: float = 1.0e-3,
            generator: Optional[torch.Generator] = None,
    ):
        super().__init__(sim_cfg)
        if initial_speckle < 0:
            raise ValueError(f"Initial speckle must be non-negative, got {initial_speckle}")
        self.initial_speckle = initial_speckle
        self.logits = nn.Parameter(self._initial_logits(generator))

    def _initial_logits(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Logits for a uniform object plus a bit of speckle to break symmetry"""
        noise = torch.randn(self.shape, generator=generator, dtype=self.sim_cfg.ftype)
        return noise * self.initial_speckle

    @property
    def field(self) -> torch.Tensor:
        """The current [H, W] object distribution, non-negative with unity sum"""
        field = torch.sigmoid(self.logits)
        return field / field.sum()

    def sample(
            self,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Returns the current field, repeated to [B, 1, H, W].
        Deterministic given the fitted logits, so `generator` is unused.
        """
        return self._repeat_static(self.field, batch_size)


class PriorSampledObject(ObjectDistribution):
    """TODO - will draw objects from a learned generative prior, if I get around to it"""
    ...
