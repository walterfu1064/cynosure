"""
Classes for managing aberration coefficients.

Includes handling the layout, bookkeeping, scaling, and whitening.

Note the two scales: `prior_scales` as the RMS that the coefs originally sample from, `target_scales`
as the priors widened by z-jitter if present, so whitened targets stay of unity order
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..config import PriorConfig
from ..zernike import ZernikeProjector


@dataclass(frozen=True)
class CoefficientBlock:
    """Represents one contiguous group of coefficients within the full vector"""
    name: str
    labels: tuple[str, ...]  # per-coefficient labels, for displaying
    prior_scales: torch.Tensor  # [N,] prior RMS for each coefficient
    pinned_mask: torch.Tensor  # [N,] bool mask of which coefficients are pinned at constants
    pinned_value: float  # value of pinned coefficients
    jitter_coupling: Optional[torch.Tensor] = None  # [N,] projection of in-medium defocus, if relevant

    @property
    def size(self) -> int:
        return len(self.labels)


def zernike_block(
        name: str,
        symbol: str,
        projector: ZernikeProjector,
        prior_cfg: PriorConfig,
        pinned_value: float,
        jitter_coupling: Optional[torch.Tensor] = None,
        ftype: torch.dtype = torch.float32,
) -> CoefficientBlock:
    """
    Builds a CoefficientBlock from a ZernikeProjector.

    Pins the piston coefficients (n = m = 0), which are arbitrary for
    phase and normalized out for intensity.
    """
    nm_indices = projector.nm_indices
    return CoefficientBlock(
        name=name,
        labels=tuple(projector.format_labels(symbol)),
        prior_scales=prior_cfg.coef_scales(nm_indices).to(ftype),
        pinned_mask=(nm_indices == 0).all(dim=1),
        pinned_value=pinned_value,
        jitter_coupling=jitter_coupling,
    )


def fit_defocus_to_phase_basis(propagator, is_phase_pinned: torch.Tensor) -> torch.Tensor:
    """
    Pre-calculates the projection of the defocus operator onto the phase Zernike
    basis so z-jitter can be represented in phase aberration space during training.

    Returns [num_phase_coefs,] per unit in-medium defocus.
    """
    mask = propagator.pupil_mask

    basis = propagator.phase_projector.zernike_bank[:, mask]  # [num_elements, num_pixels]
    if not is_phase_pinned.any():  # fit requires a piston column, append if absent
        basis = torch.cat([basis, torch.ones_like(basis[:1])], dim=0)
    waves_per_defocus = propagator.axial_wavenumber[mask] / (2 * torch.pi)

    solution = torch.linalg.lstsq(basis.T, waves_per_defocus.unsqueeze(1)).solution
    coefs = solution[:propagator.num_phase_coefs, 0]  # drop the piston column, if appended earlier
    return torch.where(is_phase_pinned, 0.0, coefs).to(propagator.ftype)


def build_aberration_space(
        propagator,
        phase_prior_cfg: PriorConfig,
        amp_prior_cfg: PriorConfig,
        z_jitter: float = 0.0,
) -> tuple['CoefficientSpace', torch.Tensor]:
    """Builds the phase+amplitude coefficient space and with the defocus decomposition for z-jittering"""
    is_phase_pinned = (propagator.phase_projector.nm_indices == 0).all(dim=1)
    defocus_phase_coefs = fit_defocus_to_phase_basis(propagator, is_phase_pinned)

    phase_block = zernike_block(
        "phase", "Z",
        propagator.phase_projector, phase_prior_cfg,
        pinned_value=0,  # pin global phase
        jitter_coupling=defocus_phase_coefs,
        ftype=propagator.ftype,
    )
    amp_block = zernike_block(
        "amp", "A",
        propagator.amp_projector, amp_prior_cfg,
        pinned_value=1,  # pin overall intensity
        ftype=propagator.ftype,
    )
    space = CoefficientSpace(
        [phase_block, amp_block],
        jitter_std=propagator.defocus_from_objective_z(z_jitter),
        ftype=propagator.ftype,
    )
    return space, defocus_phase_coefs


class CoefficientSpace(nn.Module):
    """Handles all bookkeeping, whitening, priors, etc. for the aberration coefficients"""

    def __init__(
            self,
            blocks: Sequence[CoefficientBlock],
            *,
            jitter_std: float = 0.0,
            ftype: torch.dtype = torch.float32,
    ):
        """
        Arguments:
        - blocks: ordered blocks to be catted into the ceofficient vector
        - jitter_std: z-jitter rms in in-medium defocus units, widens the whitening scales of defocus-coupled blocks
        - ftype: float dtype
        """
        super().__init__()
        if not blocks:
            raise ValueError("CoefficientSpace needs at least one CoefficientBlock")

        self.blocks = tuple(blocks)
        self.ftype = ftype
        self.jitter_std = jitter_std

        self._register_layout()
        self._register_statistics()

    def _register_layout(self) -> None:
        """Registers the pinned masks and the block sizes, keeping the block order"""
        for block in self.blocks:
            self.register_buffer(f"is_{block.name}_pinned", block.pinned_mask)
        self.register_buffer("is_pinned", torch.cat([block.pinned_mask for block in self.blocks], dim=0))

        self.block_sizes: tuple[int, ...] = tuple(block.size for block in self.blocks)
        self.num_coefs: int = sum(self.block_sizes)
        self.num_nonpinned_coefs: int = int((~self.is_pinned).sum())

        # matmul form of the scatter so masking can be done without an in-place write
        self.register_buffer(
            "nonpinned_basis",
            torch.eye(self.num_coefs, dtype=self.ftype)[:, ~self.is_pinned],
            persistent=False,
        )

    def _register_statistics(self) -> None:
        """
        Registers the prior scales and the affine whitening.
        Pinned coefs take their pinned value as their mean, so should whiten to zero (see `scatter_nonpinned`).
        Coefs coupled to defocus have their scales widened to account for z-jitter, if present.
        """
        prior_scales, target_means, target_scales = [], [], []
        jitter_variance = self.jitter_std ** 2 / 3  # variance of a uniform over +/- jitter_std

        for block in self.blocks:
            scales = block.prior_scales.to(self.ftype)
            prior_scales.append(scales)
            target_means.append(block.pinned_mask.to(self.ftype) * block.pinned_value)

            if block.jitter_coupling is not None and self.jitter_std > 0:
                widened = torch.sqrt(scales**2 + jitter_variance * block.jitter_coupling**2)
                target_scales.append(widened)
            else:
                target_scales.append(scales)

        self.register_buffer("prior_scales", torch.cat(prior_scales, dim=0), persistent=False)
        self.register_buffer("target_means", torch.cat(target_means, dim=0))
        self.register_buffer("target_scales", torch.cat(target_scales, dim=0))

    # Bookkeeping

    @property
    def coefficient_labels(self) -> list[str]:
        """Labels for all [N_tot,] coefficients, in block order"""
        return [label for block in self.blocks for label in block.labels]

    @property
    def nonpinned_labels(self) -> list[str]:
        """`coefficient_labels`, restricted to the [N_kept,] non-pinned coefficients"""
        return [
            label for label, pinned in zip(self.coefficient_labels, self.is_pinned.tolist())
            if not pinned
        ]

    def _block_index(self, name: str) -> int:
        """Returns the index of a named block in the block list"""
        for index, block in enumerate(self.blocks):
            if block.name == name:
                return index
        raise KeyError(f"No coefficient block named {name!r}, found {[b.name for b in self.blocks]}")

    def block_size(self, name: str) -> int:
        """Number of coefficients in a named block"""
        return self.blocks[self._block_index(name)].size

    def num_nonpinned_in(self, name: str) -> int:
        """Number of non-pinned coefficients in a named block"""
        return int((~self.blocks[self._block_index(name)].pinned_mask).sum())

    def join(self, *block_coefs: torch.Tensor) -> torch.Tensor:
        """Cats per-block [..., N_b] coefficients into [..., N_tot]"""
        if len(block_coefs) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} blocks, got {len(block_coefs)}")
        return torch.cat(block_coefs, dim=-1)

    def split(self, coefs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Splits a [..., N_tot] tensor into its per-block parts"""
        return torch.split(coefs, list(self.block_sizes), dim=-1)

    def gather_nonpinned(self, coefs: torch.Tensor) -> torch.Tensor:
        """Selects the [..., N_kept] non-pinned coefficients from the full [..., N_tot]"""
        return coefs[..., ~self.is_pinned]

    def scatter_nonpinned(self, kept_coefs: torch.Tensor) -> torch.Tensor:
        """Scatters [..., N_kept] non-pinned coefficients back into a full [..., N_tot] whitened vector"""
        return kept_coefs @ self.nonpinned_basis.T

    # Whitening

    def whiten(self, coefs: torch.Tensor) -> torch.Tensor:
        """Whitens a [..., N_tot] full coefficient vector into a regression target"""
        return ((coefs - self.target_means) / self.target_scales).to(self.ftype)

    def unwhiten(self, whitened: torch.Tensor) -> torch.Tensor:
        """Inverse of `whiten`, back to physical coefficients"""
        return whitened * self.target_scales + self.target_means

    def whiten_blocks(self, *block_coefs: torch.Tensor) -> torch.Tensor:
        """Joins per-block [B, N_b] coefficients and whitens them into a [B, N_tot] regression target"""
        return self.whiten(self.join(*block_coefs))

    def unwhiten_to_blocks(self, whitened: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Unwhitens a [..., N_tot] estimate and splits it into physical per-block coefficients"""
        return self.split(self.unwhiten(whitened))

    # Sampling

    def sample(
            self,
            batch_size: int,
            device: Optional[torch.device] = None,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, ...]:
        """Samples per-block [B, N_b] coefficients from the priors, holding the pinned ones fixed"""
        samples = []
        for block, scales, pinned in zip(self.blocks, self.split(self.prior_scales), self.split(self.is_pinned)):
            coefs = torch.randn(batch_size, block.size, generator=generator, dtype=self.ftype, device=device)
            coefs = coefs * scales.to(device)
            samples.append(torch.where(pinned.to(device), block.pinned_value, coefs))
        return tuple(samples)
