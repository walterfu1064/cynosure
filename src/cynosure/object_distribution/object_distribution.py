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

from ..config import BlobPriorConfig, SimulationConfig
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

    @property
    def extent(self) -> float:
        """Extent of the object grid, in physical units"""
        return self.size * self.sim_cfg.object_pixel_size

    @property
    def num_params(self) -> int:
        """
        Number of sampled parameters this distribution reports as training labels.
        Zero unless the subclass draws its objects stochastically and overrides this.
        """
        return 0

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


class SupergaussianBlobs(ObjectDistribution):
    """
    Abstract shared machinery for objects composed of supergaussian blobs.

    Holds the coordinate grid and the blob rasterizer that `KBlobs` (fitted) and
    `SampledKBlobs` (prior-sampled) both draw on.
    """

    def __init__(self, sim_cfg: SimulationConfig):
        super().__init__(sim_cfg)
        coords = (
            (torch.arange(self.size, dtype=self.sim_cfg.ftype) - self.size // 2)
            * self.sim_cfg.object_pixel_size
        )
        self.register_buffer("coords", coords, persistent=False)  # for rendering, in physical units

    def _render(
            self,
            row_positions: torch.Tensor,
            col_positions: torch.Tensor,
            diameters: torch.Tensor,
            amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rasterizes the blobs and combines them with the probabilistic union.

        Argument are each [..., K].
        Returns [..., H, W] normalized to unity sum.
        """
        dy = self.coords - row_positions[..., None]  # [..., K, H]
        dx = self.coords - col_positions[..., None]  # [..., K, W]
        r_squared = dy[..., :, None].square() + dx[..., None, :].square()  # [..., K, H, W]
        half_width_squared = (diameters[..., None, None] / 2).square()
        blobs = amplitudes[..., None, None] * torch.exp(
            -math.log(2) * (r_squared / half_width_squared).square()
        )  # supergaussian exp(-x^4)
        field = 1 - torch.prod(1 - blobs, dim=-3)  # [..., H, W]
        return field / field.sum(dim=(-2, -1), keepdim=True)


class KBlobs(SupergaussianBlobs):
    """
    An extended object formed by K fitted supergaussian blobs, each with its own
    amplitude (within [0, 1]), diameter (defined as FWHM), and position on the object grid.

    Blobs are rendered into a single object distribution via the probabilistic union,
    which is then normalized to unity sum to match the unity-pinned amplitude piston.
    """
    def __init__(
            self,
            sim_cfg: SimulationConfig,
            num_blobs: int,
            initial_diameter: float = 1.0,
            initial_spread: float = 1.0,
            rng: Optional[int | torch.Generator] = None,
    ):
        super().__init__(sim_cfg)
        if num_blobs < 1:
            raise ValueError(f"`KBlobs` requires at least one blob, got {num_blobs}")
        if initial_diameter <= 0:
            raise ValueError(f"Initial diameter must be positive, got {initial_diameter}")
        if initial_spread < 0:
            raise ValueError(f"Initial spread must be non-negative, got {initial_spread}")

        self.num_blobs = num_blobs
        self.initial_diameter = initial_diameter
        self.initial_spread = initial_spread
        if isinstance(rng, int):
            rng = torch.Generator().manual_seed(rng)

        self.amplitude_logits = nn.Parameter(torch.zeros(num_blobs, dtype=self.sim_cfg.ftype))
        self.diameter_logits = nn.Parameter(torch.zeros(num_blobs, dtype=self.sim_cfg.ftype))
        self.row_position_logits = nn.Parameter(self._initial_position_logits(rng))
        self.col_position_logits = nn.Parameter(self._initial_position_logits(rng))

    def _initial_position_logits(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Creates random position logits with a given spread for an asymmetric initialization"""
        noise = torch.randn(self.num_blobs, generator=generator, dtype=self.sim_cfg.ftype)
        return noise * self.initial_spread

    @property
    def amplitudes(self) -> torch.Tensor:
        """Converts the amplitude logits to actual amplitudes"""
        return torch.sigmoid(self.amplitude_logits)

    @property
    def diameters(self) -> torch.Tensor:
        """Converts the diameter logits to FWHM diameters, in physical units"""
        return self.initial_diameter * torch.exp(self.diameter_logits)

    @property
    def row_positions(self) -> torch.Tensor:
        """Converts the row position logits to physical-unit offsets from the grid center"""
        return (torch.sigmoid(self.row_position_logits) - 0.5) * self.extent

    @property
    def col_positions(self) -> torch.Tensor:
        """Converts the column position logits to physical-unit offsets from the grid center"""
        return (torch.sigmoid(self.col_position_logits) - 0.5) * self.extent

    @property
    def field(self) -> torch.Tensor:
        """Returns the currently rendered [H, W] object distribution"""
        return self._render(self.row_positions, self.col_positions, self.diameters, self.amplitudes)

    def sample(
            self,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Returns the current union of blobs, repeated to [B, 1, H, W].
        Deterministic given the fitted blob parameters, so `generator` is unused.
        """
        return self._repeat_static(self.field, batch_size)


class SampledKBlobs(SupergaussianBlobs):
    """
    An extended object formed by up to K supergaussian blobs with prior-sampled parameters.
    The stochastic sibling of `KBlobs`.

    Instead of holding fitted parameters, provides functions for sampling fresh blob
    parameters for each element in a batch, so a decoder can be trained on its outputs.
    The parameters are sampled from an unconstrained label space with a Gaussian prior,
    set by the `BlobPriorConfig`.

    Blobs are canonically ordered by descending amplitude, with the brightest pinned to
    exactly 1 and its logit dropped from the labels (since absolute brightness will get
    pinned by the unit-sum normalization).
    """
    def __init__(
            self,
            sim_cfg: SimulationConfig,
            num_blobs: int,
            prior_cfg: BlobPriorConfig,
    ):
        super().__init__(sim_cfg)
        if num_blobs < 1:
            raise ValueError(f"`SampledKBlobs` requires at least one blob, got {num_blobs}")
        self.num_blobs = num_blobs
        self.prior_cfg = prior_cfg

    @property
    def num_params(self) -> int:
        """{amplitude, row, column, diameter} = 4 dofs per blob, minus 1 for the pinned max amplitude"""
        return 4 * self.num_blobs - 1

    @property
    def param_labels(self) -> list[str]:
        """Human-readable names for the label vector entries, in label order"""
        return (
                [f"row_{k}" for k in range(self.num_blobs)]
                + [f"col_{k}" for k in range(self.num_blobs)]
                + [f"log_diam_{k}" for k in range(self.num_blobs)]
                + [f"amp_logit_{k}" for k in range(1, self.num_blobs)]
        )

    @property
    def param_scales(self) -> torch.Tensor:
        """Per-parameter prior RMS, for whitening the labels"""
        return self.prior_cfg.param_scales(self.num_blobs).to(self.coords)

    def sample_params(
            self,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Draws a [B, 4K - 1] batch of blob params, ordered as [rows..., cols..., diams..., amps_sans_one...].
        Blobs are canonically sorted by descending amplitude logit.
        """
        draws = torch.randn(
            batch_size, self.num_params,
            generator=generator, dtype=self.sim_cfg.ftype, device=self.coords.device,
        )
        K = self.num_blobs
        params = draws * self.param_scales
        amp_logits, _ = torch.sort(params[..., 3*K:], dim=-1, descending=True)
        return torch.cat([params[..., :3*K], amp_logits], dim=-1)

    def params_to_blobs(
            self,
            params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Maps [..., 4K - 1] label-space parameters to physical blob parameters for `_render`.
        """
        K = self.num_blobs
        rows = params[..., :K]
        cols = params[..., K:2*K]
        diameters = self.prior_cfg.reference_diameter * torch.exp(params[..., 2*K:3*K])
        pinned = torch.ones_like(params[..., :1])  # the brightest blob's amplitude
        amplitudes = torch.cat([pinned, torch.sigmoid(params[..., 3*K:])], dim=-1)
        return rows, cols, diameters, amplitudes

    def sample_with_params(
            self,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draws fresh blob parameters and renders them.
        Returns [B, 1, H, W] rendered objects and their [B, 4K - 1] label-space parameters.
        """
        params = self.sample_params(batch_size, generator)
        field = self._render(*self.params_to_blobs(params))
        return field.unsqueeze(1), params

    def sample(
            self,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Returns a [B, 1, H, W] batch of independently drawn objects.
        """
        return self.sample_with_params(batch_size, generator)[0]


### Setting aside the below for now, since on reflection, it's a really bad parameterization.
#
# class FreeField(ObjectDistribution):
#     """
#     An unconstrained, fittable object distribution.
#
#     Represents the object as a tensor of logits on the object grid.
#     Sigmoiding and then normalizing to unity sum (to match the pinned amplitude piston)
#     yields the object distribution itself, for convolving with the PSF.
#     """
#     def __init__(
#             self,
#             sim_cfg: SimulationConfig,
#             initial_speckle: float = 1.0e-3,
#             rng: Optional[int | torch.Generator] = None,
#     ):
#         super().__init__(sim_cfg)
#         if initial_speckle < 0:
#             raise ValueError(f"Initial speckle must be non-negative, got {initial_speckle}")
#         self.initial_speckle = initial_speckle
#         if isinstance(rng, int):
#             rng = torch.Generator().manual_seed(rng)
#         self.logits = nn.Parameter(self._initial_logits(rng))
#
#     def _initial_logits(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
#         """Logits for a uniform object plus a bit of speckle to break symmetry"""
#         noise = torch.randn(self.shape, generator=generator, dtype=self.sim_cfg.ftype)
#         return noise * self.initial_speckle
#
#     @property
#     def field(self) -> torch.Tensor:
#         """The current [H, W] object distribution, non-negative with unity sum"""
#         field = torch.sigmoid(self.logits)
#         return field / field.sum()
#
#     def sample(
#             self,
#             batch_size: int = 1,
#             generator: Optional[torch.Generator] = None,
#     ) -> torch.Tensor:
#         """
#         Returns the current field, repeated to [B, 1, H, W].
#         Deterministic given the fitted logits, so `generator` is unused.
#         """
#         return self._repeat_static(self.field, batch_size)


# class PriorSampledObject(ObjectDistribution):
#     """TODO - will draw objects from a learned generative prior, if I get around to it"""
#     ...
