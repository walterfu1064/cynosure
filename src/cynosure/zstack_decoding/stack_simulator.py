"""
Handles forwards simulation of aberrated z-stacks, and the synthesis of labeled training data.

Owns the physical parts of a solver: BeamPropagator, ObjectDistribution, NoiseModel, and
the z-positions of the image plane(s).
"""

from typing import Optional

import torch
import torch.nn as nn

from .coefficients import CoefficientSpace, build_aberration_space
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
            z_objective: Optional[torch.Tensor] = None,
            z_jitter: float = 0.0,
            noise_cfg: Optional[NoiseConfig] = None,
            chunk_size: Optional[int] = None,
    ):
        """
        Arguments:
        - propagator: beam propagation simulator
        - object_distribution: object being imaged, to be convolved with the PSF
        - coefficients: layout and priors for the aberration coefficients
        - defocus_phase_coefs: [num_phase,] projection of in-medium defocus, from `fit_defocus_to_phase_basis`
        - z_objective: physical objective positions the stack is taken at
        - z_jitter: half-width of the uniform rigid z-offset applied to each synthetic stack
        - noise_cfg: if given, NoiseModel args, used to apply noise before normalization
        - chunk_size: stacks per propagation chunk, to bound peak memory
        """
        super().__init__()
        self.propagator = propagator
        self.object_distribution = object_distribution
        self.coefficients = coefficients
        self.noise_model = NoiseModel(noise_cfg) if noise_cfg else None
        self.noise_cfg = noise_cfg
        self.ftype = propagator.ftype
        self.chunk_size = chunk_size

        if z_objective is None:
            z_objective = torch.full((1,), fill_value=z_jitter, dtype=self.ftype)
        self.register_buffer("z_objective", z_objective.to(self.ftype))
        self.register_buffer("defocus_phase_coefs", defocus_phase_coefs)
        self.num_z = z_objective.shape[0]
        self.z_jitter = float(z_jitter)  # plain float: consumers do ordinary arithmetic on it

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
            z_objective: Optional[torch.Tensor] = None,
            z_jitter: float = 0.0,
            noise_cfg: Optional[NoiseConfig] = None,
            chunk_size: Optional[int] = None,
    ) -> 'StackSimulator':
        """Builds a forwards simulator (including its propagator and coefficient space) from configs"""
        propagator = BeamPropagator(sim_cfg, optics_cfg, phase_cfg, amp_cfg=amp_cfg)
        coefficients, defocus_phase_coefs = build_aberration_space(
            propagator, phase_prior_cfg, amp_prior_cfg, z_jitter
        )
        return cls(
            propagator=propagator,
            object_distribution=object_distribution,
            coefficients=coefficients,
            defocus_phase_coefs=defocus_phase_coefs,
            z_objective=z_objective,
            z_jitter=z_jitter,
            noise_cfg=noise_cfg,
            chunk_size=chunk_size,
        )

    @property
    def device(self) -> torch.device:
        """Using `z_objective` as a proxy"""
        return self.z_objective.device

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
    ) -> torch.Tensor:
        """
        Forwards-simulates a z-stack (or a batch of such) from the given defocus values and aberrations.
        Propagation happens in chunks to bound memory use.

        The trailing dimension of `z` is taken to be the z-stack size, which must match `self.num_z`.
        In general, `z` should be [B, Z] (with singlet dimensions as needed). A one-dimensional [B,]
        is only accepted when `self.num_z` is 1.

        This is all a consequence of BeamPropagator not differentiating between batches, z-stacks,
        and batches of z-stacks, instead folding all non-spatial dimensions into the batch dimension.

        Arguments:
        - z: defocus values from `defocus_from_objective_z`, either [B,] or [B, Z]
        - phase_coefs: phase coefficients, [B, N_phase]
        - amp_coefs: amplitude coefficients, [B, N_amp]
        """
        if z.ndim == 1 and self.num_z == 1:
            z = z.unsqueeze(-1)  # [B,] -> [B, 1]
        if z.ndim != 2 or z.shape[-1] != self.num_z:
            raise ValueError(f"`z` must be [B, num_z] with num_z = {self.num_z}, got {tuple(z.shape)}")
        if not phase_coefs.shape[0] == amp_coefs.shape[0] == z.shape[0]:
            raise ValueError(
                "Got different batch dims for `z`, `phase_coefs`, and `amp_coefs`: "
                f"{z.shape[0]}, {phase_coefs.shape[0]}, {amp_coefs.shape[0]}"
            )

        batch_size = z.shape[0]
        chunk_size = self.chunk_size or batch_size
        image_chunks = []
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            z_chunk = z[start:stop].reshape(-1)
            phase_chunk = phase_coefs[start:stop].repeat_interleave(self.num_z, dim=0)
            amp_chunk = amp_coefs[start:stop].repeat_interleave(self.num_z, dim=0)
            psf = self.propagator(z_chunk, phase_chunk, amp_chunk)
            psf = psf.reshape(stop - start, self.num_z, *psf.shape[-2:])  # unfold to [B, Z, H, W]
            img = self.object_distribution(psf, batch_size=stop - start)
            image_chunks.append(img)
        return torch.cat(image_chunks)

    def sample_z_jitter(
            self,
            batch_size: int,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Samples [B,] objective-z defocus offsets uniformly over [-z_jitter, z_jitter].
        If `z_jitter` is 0, returns zeros.
        """
        if self.z_jitter == 0:
            return torch.zeros(batch_size, dtype=self.ftype, device=self.device)
        uniform = torch.rand(batch_size, generator=generator, dtype=self.ftype, device=self.device)
        return (uniform * 2 - 1) * self.z_jitter

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
    ) -> torch.Tensor:
        """
        Returns the [B, num_z] in-medium defocus corresponding to the stored z positions.
        Rigidly shifts each z-stack by `offsets` (in objective-z units) if given.
        """
        z = self.z_objective.unsqueeze(0).expand(batch_size, self.num_z)
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
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Forwards-simulates the normalized z-stacks at the model's own z positions
        from physical [B, N] coefficients (e.g. as returned by `predict_coefficients`).
        If `with_noise`, adds noise before normalizing.
        If `offsets` is given, rigidly shifts each z-stack, preserving the relative z-spacing.

        Returns [B, num_z, H, W].
        """
        with torch.no_grad():
            z = self.batched_defocus(phase_coefs.shape[0], offsets=offsets)
            images = self.simulate_stacks(z, phase_coefs, amp_coefs)
            if with_noise:
                images = self.apply_noise(images, generator)
            return self.normalize_stack(images).float()

    def create_examples(
            self,
            batch_size: int,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a set of aberrations, forwards-simulates the z-stacks, corrupts
        them with the noise model, and normalizes them for use as inputs to the CNN.

        If `z_jitter` is set, calculates the synthetic z-stacks under random rigid offsets,
        and adds the defocus-equivalent phase shifts to the synthetic labels.

        Returns:
        - images: [B, num_z, H, W] normalized z-stacks
        - phase_coefs: [B, num_phase_coefs] effective phase aberration coefficients (including jitter offsets)
        - amp_coefs: [B, num_amp_coefs] amp aberration coefficients
        """
        phase_coefs, amp_coefs = self.sample_coefficients(batch_size, generator=generator)
        offsets = self.sample_z_jitter(batch_size, generator=generator)
        images = self.simulate_normalized_stacks(
            phase_coefs, amp_coefs, with_noise=True, offsets=offsets, generator=generator
        )
        phase_coefs = phase_coefs + self.z_offset_to_coefficients(offsets)
        return images, phase_coefs, amp_coefs
