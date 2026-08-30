"""
Methods of choosing the nominal z-stack geometry, i.e. the objective z-positions
the planes of each stack are taken at.

A GeometryMode draws the plane count once per `sample` call, so every batch stays a
dense rectangular tensor and no padding or masking is ever needed; geometry diversity
comes from resampling across batches. The plane *positions* may still vary per stack
within a batch.

All numbers are in physical objective units.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


class GeometryMode(ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def fixed_planes(self) -> Optional[torch.Tensor]:
        """The shared [Z,] z-positions when the geometry is fully fixed, or None when it varies"""
        ...

    @property
    def num_z(self) -> Optional[int]:
        """
        The plane count when the geometry is fully fixed, or None when it varies.

        Deliberately None even for modes with a constant plane count but varying positions:
        a geometry-blind (CNN) trunk couldn't tell such stacks apart, so only fully fixed
        geometries should report a count for it to bake in.
        """
        planes = self.fixed_planes
        return None if planes is None else int(planes.shape[0])

    @abstractmethod
    def sample(
            self,
            batch_size: int,
            *,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Returns a [batch_size, Z] tensor of nominal z-positions, drawing Z once per call"""
        ...


class FixedGeometry(GeometryMode):
    """Every stack is taken at the same fixed z-positions"""

    __slots__ = ("z_objective",)

    def __init__(self, z_objective: torch.Tensor):
        if z_objective.ndim != 1 or z_objective.numel() == 0:
            raise ValueError(f"z_objective must be a non-empty [Z,] tensor, got {tuple(z_objective.shape)}")
        self.z_objective = z_objective

    @property
    def fixed_planes(self) -> torch.Tensor:
        return self.z_objective

    def sample(
            self,
            batch_size: int,
            *,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        z = self.z_objective.to(device=device, dtype=dtype)
        return z.unsqueeze(0).expand(batch_size, -1)


@dataclass(frozen=True, slots=True)
class UniformSpanGeometry(GeometryMode):
    """
    Evenly-spaced stacks, symmetric about z=0, with randomized plane count and axial extent.

    The plane count is drawn uniformly over [min_planes, max_planes] once per call, then
    each stack draws its own half-span uniformly over [min_half_span, max_half_span] and
    spaces its planes evenly across [-half_span, +half_span].
    """

    min_planes: int
    max_planes: int
    min_half_span: float
    max_half_span: float

    def __post_init__(self):
        if not 2 <= self.min_planes <= self.max_planes:
            raise ValueError(
                f"Require 2 <= min_planes <= max_planes, got {self.min_planes} and {self.max_planes}"
            )
        if not 0 < self.min_half_span <= self.max_half_span:
            raise ValueError(
                f"Require 0 < min_half_span <= max_half_span, got {self.min_half_span} and {self.max_half_span}"
            )

    @property
    def fixed_planes(self) -> None:
        return None

    def sample(
            self,
            batch_size: int,
            *,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        num_planes = torch.randint(
            self.min_planes,
            self.max_planes + 1,
            (1,),
            generator=generator,
            device=device,
        ).item()
        ladder = torch.linspace(-1, 1, int(num_planes), dtype=dtype, device=device)  # [Z,]
        half_spans = self.min_half_span + (self.max_half_span - self.min_half_span) * torch.rand(
            batch_size, 1, generator=generator, dtype=dtype, device=device,
        )  # [B, 1]
        return half_spans * ladder  # [B, Z]


def as_geometry_mode(geometry: torch.Tensor | GeometryMode) -> GeometryMode:
    """Passes a GeometryMode through, promotes a plain [Z,] tensor to `FixedGeometry`"""
    if isinstance(geometry, GeometryMode):
        return geometry
    return FixedGeometry(geometry)


# Frozen holders of primitives (or, for FixedGeometry, a tensor), safe for `torch.load`
torch.serialization.add_safe_globals([FixedGeometry, UniformSpanGeometry])
