"""
Handles forwards simulation of aberrated z-stacks, and the synthesis of labeled training data.

Owns the physical parts of a solver: BeamPropagator, ObjectDistribution, NoiseModel, and
the GeometryMode describing the z-positions the stacks are taken at.

FixedGeometry simulators take the z-planes as a `z_objective` buffer, sampled geometries draw
fresh nominal positions per batch.
"""

from typing import Optional

import torch
import torch.nn as nn

from .coefficients import CoefficientSpace, build_coefficient_space
from .geometry_modes import GeometryMode, as_geometry_mode
from .jitter_modes import JitterMode, NoJitter, as_jitter_mode
from .noise_model import NoiseModel
from ..beam_propagation import BeamPropagator
from ..config import NoiseConfig, OpticalConfig, PriorConfig, SimulationConfig, ZernikeConfig
from ..object_distribution import ObjectDistribution


class StackSimulator(nn.Module):
    """Synthesizes aberrated z-stacks"""

    def __init__(
            self,
            *,
            propagator: BeamPropagator,
            object_distribution: ObjectDistribution,
            coefficients: CoefficientSpace,
            defocus_phase_coefs: torch.Tensor,
            z_objective: Optional[torch.Tensor | GeometryMode] = None,
            z_jitter: float | JitterMode = NoJitter(),
            noise_cfg: Optional[NoiseConfig] = None,
            chunk_size: Optional[int] = None,
    ):
        """
        Since the BeamPropagator and CoefficientSpace are fully defined by the configs and
        must be kept in sync with them, StackSimulator should usually be instantiated via
        `from_configs`, not from this method.

        Arguments:
        - propagator: beam propagation simulator
        - object_distribution: object being imaged, to be convolved with the PSF
        - coefficients: layout and priors for the aberration coefficients
        - defocus_phase_coefs: [num_phase,] projection of in-medium defocus, from `fit_defocus_to_phase_basis`
        - z_objective: physical objective positions or config the stack is taken at (defaults to one-sided jitter)
        - z_jitter: distribution of the rigid z-offset applied to each synthetic stack
        - noise_cfg: if given, NoiseModel args, used to apply noise before normalization
        - chunk_size: stacks per propagation chunk, to bound peak memory
        """
        super().__init__()
        self.propagator = propagator
        self.object_distribution = object_distribution
        self.coefficients = coefficients
        self.noise_model: NoiseModel = NoiseModel(noise_cfg) if noise_cfg else None
        self.noise_cfg = noise_cfg
        self.ftype = propagator.ftype
        self.chunk_size = chunk_size
        self.z_jitter: JitterMode = as_jitter_mode(z_jitter)

        if z_objective is None:  # place the single plane so the jitter only ever reaches one side of focus
            z_objective = torch.full((1,), fill_value=self.z_jitter.max_offset, dtype=self.ftype)
        self.geometry: GeometryMode = as_geometry_mode(z_objective)
        planes = self.geometry.fixed_planes  # None if stack geometry is sampled per batch
        self.register_buffer("z_objective", None if planes is None else planes.to(self.ftype))
        self.register_buffer("defocus_phase_coefs", defocus_phase_coefs)
        self.num_z = self.geometry.num_z

    @classmethod
    def from_configs(
            cls,
            *,
            sim_cfg: SimulationConfig,
            optics_cfg: OpticalConfig,
            phase_cfg: ZernikeConfig,
            amp_cfg: ZernikeConfig,
            phase_prior_cfg: PriorConfig,
            amp_prior_cfg: PriorConfig,
            object_distribution: ObjectDistribution,
            z_objective: Optional[torch.Tensor | GeometryMode] = None,
            z_jitter: float | JitterMode = NoJitter(),
            noise_cfg: Optional[NoiseConfig] = None,
            chunk_size: Optional[int] = None,
    ) -> 'StackSimulator':
        """Builds a forwards simulator (including its propagator and coefficient space) from configs"""
        jitter = as_jitter_mode(z_jitter)  # shared, so the whitening always matches what gets sampled
        propagator = BeamPropagator(sim_cfg, optics_cfg, phase_cfg, amp_cfg=amp_cfg)
        coefficients, defocus_phase_coefs = build_coefficient_space(
            propagator, phase_prior_cfg, amp_prior_cfg, jitter, object_distribution=object_distribution,
        )
        return cls(
            propagator=propagator,
            object_distribution=object_distribution,
            coefficients=coefficients,
            defocus_phase_coefs=defocus_phase_coefs,
            z_objective=z_objective,
            z_jitter=jitter,
            noise_cfg=noise_cfg,
            chunk_size=chunk_size,
        )

    @property
    def device(self) -> torch.device:
        """Using `defocus_phase_coefs` as a proxy"""
        return self.defocus_phase_coefs.device

    def sample_coefficients(
            self,
            batch_size: int,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        Samples per-block [B, N_b] coefficients from the priors.
        Thin wrapper over `CoefficientSpace.sample` to pass the device info.
        """
        return self.coefficients.sample(batch_size, device=self.device, generator=generator)

    @staticmethod
    def normalize_stack(images: torch.Tensor) -> torch.Tensor:
        """Roughly normalizes a [..., H, W] image stack per-plane/-batch element"""
        images = images - images.amin((-2, -1), keepdim=True)
        sums = images.sum((-2, -1), keepdim=True)
        return images / sums

    def simulate_stacks(
            self,
            z: torch.Tensor,
            phase_coefs: torch.Tensor,
            amp_coefs: torch.Tensor,
            objects: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forwards-simulates a z-stack (or a batch of such) from the given defocus values and aberrations.
        Propagation happens in chunks to bound memory use.

        The trailing dimension of `z` is taken to be the z-stack size.
        This method is geometry-agnostic any [B, Z] is accepted, regardless of the GeometryMode.
        A one-dimensional [B,] is read as a batch of single planes.

        This is all a consequence of BeamPropagator not differentiating between batches, z-stacks,
        and batches of z-stacks, instead folding all non-spatial dimensions into the batch dimension.

        Arguments:
        - z: defocus values from `defocus_from_objective_z`, either [B,] or [B, Z]
        - phase_coefs: phase coefficients, [B, N_phase]
        - amp_coefs: amplitude coefficients, [B, N_amp]
        - objects: optional pre-drawn [B, 1, H, W] objects; freshly sampled per chunk when omitted
        - generator: for internal object draws, unused for non-stochastic objects
        """
        if z.ndim == 1:
            z = z.unsqueeze(-1)  # [B,] -> [B, 1]
        if z.ndim != 2:
            raise ValueError(f"`z` must be [B, Z] or [B,], got {tuple(z.shape)}")
        num_z = z.shape[-1]
        if not phase_coefs.shape[0] == amp_coefs.shape[0] == z.shape[0]:
            raise ValueError(
                "Got different batch dims for `z`, `phase_coefs`, and `amp_coefs`: "
                f"{z.shape[0]}, {phase_coefs.shape[0]}, {amp_coefs.shape[0]}"
            )
        if objects is not None and objects.shape[0] != z.shape[0]:
            raise ValueError(f"Batch dims differ for `objects` and `z`: {objects.shape[0]}, {z.shape[0]}")

        batch_size = z.shape[0]
        chunk_size = chunk_size if chunk_size is not None else (self.chunk_size or batch_size)
        image_chunks = []
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            z_chunk = z[start:stop].reshape(-1)
            phase_chunk = phase_coefs[start:stop].repeat_interleave(num_z, dim=0)
            amp_chunk = amp_coefs[start:stop].repeat_interleave(num_z, dim=0)
            psf = self.propagator(z_chunk, phase_chunk, amp_chunk)
            psf = psf.reshape(stop - start, num_z, *psf.shape[-2:])  # unfold to [B, Z, H, W]
            if objects is None:
                img = self.object_distribution(psf, batch_size=stop - start, generator=generator)
            else:
                img = self.object_distribution.convolve_psf(psf, objects[start:stop])
            image_chunks.append(img)
        return torch.cat(image_chunks)

    def sample_z_jitter(
            self,
            batch_size: int,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Samples [B,] rigid objective-z offsets from the jitter mode"""
        return self.z_jitter.sample(batch_size, device=self.device, dtype=self.ftype, generator=generator)

    def sample_geometry(
            self,
            batch_size: int,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Samples [B, Z] nominal objective z-positions from the geometry mode (Z drawn per call)"""
        return self.geometry.sample(batch_size, device=self.device, dtype=self.ftype, generator=generator)

    def z_offset_to_coefficients(self, z_offsets: torch.Tensor) -> torch.Tensor:
        """
        Returns the [..., num_phase] aberrations equivalent to the [...,] z-offsets (in objective-z units).

        Synthetic images will be calculated using the z-offsets but not this phase, while
        their phase coef labels will include this phase in place of the z-offsets.
        """
        defocus = self.propagator.defocus_from_objective_z(z_offsets)
        return defocus.unsqueeze(1) * self.defocus_phase_coefs

    def batched_defocus(
            self,
            batch_size: int,
            offsets: Optional[torch.Tensor] = None,
            z_objective: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns the [B, Z] in-medium defocus corresponding to the given z positions,
        defaulting to the stored fixed geometry.
        Rigidly shifts each z-stack by `offsets` (in objective-z units) if given.

        Arguments:
        - offsets: optional [B,] rigid shifts, e.g. from `sample_z_jitter`
        - z_objective: [Z,] or [B, Z] nominal objective positions (required for non-fixed geometries)
        """
        if z_objective is None:
            z_objective = self.z_objective
            if z_objective is None:
                raise ValueError("Geometry isn't fixed, so `z_objective` must be passed explicitly")
        if z_objective.ndim == 1:
            z_objective = z_objective.unsqueeze(0)
        z = z_objective.expand(batch_size, z_objective.shape[-1])
        if offsets is not None:
            z = z + offsets.unsqueeze(1)
        return self.propagator.defocus_from_objective_z(z)

    def apply_noise(
            self,
            images: torch.Tensor,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Adds noise to a [..., Z, H, W] image stack.

        `noise_cfg`'s photon count and background are both per-frame, so the trailing
        dims of `images` must be z-stack-shaped, [..., Z, H, W]. Use a singlet
        z-dimension [..., 1, H, W] if necessary.
        """
        if self.noise_model:
            return self.noise_model(images, generator=generator)
        return images

    def simulate_normalized_stacks(
            self,
            phase_coefs: torch.Tensor,
            amp_coefs: torch.Tensor,
            with_noise: bool = False,
            offsets: Optional[torch.Tensor] = None,
            objects: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            chunk_size: Optional[int] = None,
            z_objective: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forwards-simulates the normalized z-stacks at the given z positions
        from physical [B, N] coefficients (e.g. as returned by `predict_coefficients`).
        If None, defaults to the model's own fixed geometry.
        If `with_noise`, adds noise before normalizing.
        If `offsets` is given, rigidly shifts each z-stack, preserving the relative z-spacing.
        If `objects` is given, images those [B, 1, H, W] objects instead of fresh draws.
        If `z_objective` is given ([Z,] or [B, Z]), images at those nominal positions instead
        of the stored ones (required when the stored geometry isn't fixed).

        Returns [B, Z, H, W].
        """
        with torch.no_grad():
            z = self.batched_defocus(phase_coefs.shape[0], offsets=offsets, z_objective=z_objective)
            images = self.simulate_stacks(
                z,
                phase_coefs,
                amp_coefs,
                objects=objects,
                generator=generator,
                chunk_size=chunk_size,
            )
            if with_noise:
                images = self.apply_noise(images, generator)
            return self.normalize_stack(images).float()

    def create_examples(
            self,
            batch_size: int,
            generator: Optional[torch.Generator] = None,
            chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        Samples a set of aberrations, a stack geometry (if stochastic), and an object
        per stack (if stochastic), forwards-simulates the z-stacks, corrupts them
        with the noise model, and normalizes them for use as inputs to the network.

        If `z_jitter` is set, calculates the synthetic z-stacks under random rigid offsets,
        and adds the defocus-equivalent phase shifts to the synthetic labels. The returned
        `z` stays nominal so the network doesn't get to see the jitter directly.

        Returns the network inputs (`images` and their nominal z-positions), followed by
        one label tensor per block of the coefficient space:
        - images: [B, Z, H, W] normalized z-stacks
        - z: [B, Z] nominal objective z-positions the stacks were taken at
        - phase_coefs: [B, num_phase_coefs] effective phase aberration coefficients (including jitter offsets)
        - amp_coefs: [B, num_amp_coefs] amp aberration coefficients
        - object_params: [B, num_params] object labels if reported by the object distribution
        """
        phase_coefs, amp_coefs = self.sample_coefficients(batch_size, generator=generator)
        z = self.sample_geometry(batch_size, generator=generator)
        offsets = self.sample_z_jitter(batch_size, generator=generator)
        objects, object_params = self.object_distribution.sample_with_params(batch_size, generator)
        images = self.simulate_normalized_stacks(
            phase_coefs,
            amp_coefs,
            with_noise=True,
            offsets=offsets,
            objects=objects,
            generator=generator,
            chunk_size=chunk_size,
            z_objective=z,
        )
        phase_coefs = phase_coefs + self.z_offset_to_coefficients(offsets)
        if self.object_distribution.num_params:
            return images, z, phase_coefs, amp_coefs, object_params
        return images, z, phase_coefs, amp_coefs
