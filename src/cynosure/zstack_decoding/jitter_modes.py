"""
Methods of randomly jittering the z-position for an image or z-stack.
In the z-stack case, jitter is rigid across slices.
All numbers are in physical objective units.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


class JitterMode(ABC):
    __slots__ = ()

    @abstractmethod
    def sample(
            self,
            num_samples: int,
            *,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Returns a [num_samples,] tensor of independently drawn offsets"""
        ...

    @property
    @abstractmethod
    def variance(self) -> float:
        """Variance of the offsets, for widening the defocus-coupled coefficient priors"""
        ...

    @property
    @abstractmethod
    def max_offset(self) -> float:
        """Largest offset that can be drawn, e.g. for placing the nominal z-positions"""
        ...


@dataclass(frozen=True, slots=True)
class NoJitter(JitterMode):
    """Null case with zero jitter"""

    def sample(
            self,
            num_samples: int,
            *,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return torch.zeros(num_samples, dtype=dtype, device=device)

    @property
    def variance(self) -> float:
        return 0.0

    @property
    def max_offset(self) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class UniformJitter(JitterMode):
    """Uniformly distributed over [-max_jitter, +max_jitter]"""

    max_jitter: float

    def __post_init__(self):
        if self.max_jitter < 0:
            raise ValueError(f"max_jitter must be non-negative, got {self.max_jitter}")

    def sample(
            self,
            num_samples: int,
            *,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        signed = torch.rand(num_samples, generator=generator, dtype=dtype, device=device) * 2 - 1
        return signed * self.max_jitter

    @property
    def variance(self) -> float:
        return self.max_jitter ** 2 / 3

    @property
    def max_offset(self) -> float:
        return self.max_jitter


@dataclass(frozen=True, slots=True)
class ShellJitter(JitterMode):
    """Uniformly distributed over [-max_jitter, -min_jitter] U [+min_jitter, +max_jitter]"""

    min_jitter: float
    max_jitter: float

    def __post_init__(self):
        if not 0 <= self.min_jitter <= self.max_jitter:
            raise ValueError(
                f"Require 0 <= min_jitter <= max_jitter, got {self.min_jitter} and {self.max_jitter}"
            )

    def sample(
            self,
            num_samples: int,
            *,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        signed = torch.rand(num_samples, generator=generator, dtype=dtype, device=device) * 2 - 1
        magnitude = signed.abs() * (self.max_jitter - self.min_jitter) + self.min_jitter
        return torch.copysign(magnitude, signed)

    @property
    def variance(self) -> float:
        low, high = self.min_jitter, self.max_jitter
        return (low ** 2 + low * high + high ** 2) / 3

    @property
    def max_offset(self) -> float:
        return self.max_jitter


def as_jitter_mode(jitter: float | JitterMode) -> JitterMode:
    """Passes a JitterMode through, promotes a plain half-width float to `UniformJitter`"""
    if isinstance(jitter, JitterMode):
        return jitter
    return UniformJitter(float(jitter)) if jitter > 0 else NoJitter()
