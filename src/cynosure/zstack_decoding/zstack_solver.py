"""
PyTorch Lightning model that trains itself to infer phase and amplitude
aberrations from z-stacks.

Training data is synthesized on the fly using Zernike coefficients sampled
from a decaying-spectrum prior, and propagating forwards to get the synthetic images.

If a `NoiseConfig` is given, photon and camera noise are added to the synthesized
images before normalization (training/validation, not prediction).

Aberration coefficients are scaled/unscaled before/after using to train,
to keep them on an easy scale for the model to learn.

Validation patches are generated in the same way, but with a fixed seed so
they're consistent across epochs.
"""

import math
from collections.abc import Sequence
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .noise_model import NoiseModel
from .submodules import CnnDecoder, CnnDecoder_DetachedHead, VelocityField
from ..beam_propagation import BeamPropagator
from ..config import (
    NoiseConfig,
    OpticalConfig,
    PriorConfig,
    SimulationConfig,
    TrainingConfig,
    VelocityConfig,
    ZernikeConfig,
)
from ..utilities.fft_utilities import convolve_psf_with_object
from ..utilities.ode_solvers import EulerSolver, ODESolver
from ..zernike import ZernikeProjector


def make_object_distribution(
        object_grid_size: int,
        object_pixel_size: float,
        bead_diameter: float,
) -> torch.Tensor:
    coords = torch.arange(-(object_grid_size//2), object_grid_size//2 + 1) * object_pixel_size
    Y, X = torch.meshgrid(coords, coords, indexing="ij")
    R = torch.sqrt(X*X + Y*Y)
    object_distrib = (R <= bead_diameter/2).float()
    return object_distrib


class ZstackSolver(pl.LightningModule):
    """
    Module that contains an imaging system configuration and a z-stack decoder.
    Using its own generated synthetic data, tarins its decoder to predict
    the most likely set of aberration coefficients from a given z-stack.
    """
    def __init__(
            self,
            *,
            train_cfg: TrainingConfig,
            sim_cfg: SimulationConfig,
            optics_cfg: OpticalConfig,
            phase_cfg: ZernikeConfig,
            amp_cfg: ZernikeConfig,
            phase_prior_cfg: PriorConfig,
            amp_prior_cfg: PriorConfig,
            object_distribution: torch.Tensor,
            z_objective: Optional[torch.Tensor] = None,
            z_jitter: float = 0.0,
            noise_cfg: Optional[NoiseConfig] = None,
            hidden_channels: Sequence[int] = (16, 32),
            embedding_dims: int = 128,
    ):
        super().__init__()

        self.train_cfg = train_cfg
        self.sim_cfg = sim_cfg
        self.optics_cfg = optics_cfg
        self.phase_cfg = phase_cfg
        self.amp_cfg = amp_cfg
        self.phase_prior_cfg = phase_prior_cfg
        self.amp_prior_cfg = amp_prior_cfg
        self.noise_cfg = noise_cfg

        self.propagator = BeamPropagator(sim_cfg, optics_cfg, phase_cfg, amp_cfg=amp_cfg)
        self.noise_model = NoiseModel(noise_cfg) if noise_cfg is not None else None
        self.ftype = self.propagator.ftype

        if z_objective is None:
            z_objective = torch.full((1,), fill_value=z_jitter, dtype=self.ftype)
        self.register_buffer("z_objective", z_objective.to(self.ftype))
        self.num_z = z_objective.shape[0]
        self.z_jitter = z_jitter

        self._setup_piston_flags()
        self._setup_jitter_decomposition()  # must come before whitening, which widens the jittered target scales
        self._setup_whitening()  # must be initialized before decoder
        self.decoder = self._setup_decoder(hidden_channels, embedding_dims)

        self.register_buffer("object_distribution", object_distribution.to(self.ftype))

    def _setup_decoder(
            self,
            hidden_channels: Sequence[int],
            embedding_dims: int,
    ) -> nn.Module:
        if not hasattr(self, "decoder_out_dims"):
            raise RuntimeError("`decoder_out_dims` is uninitialized, call `_setup_whitening` first")
        decoder = CnnDecoder(
            in_channels=self.num_z,
            spatial_hidden_channels=hidden_channels,
            embedding_dims=embedding_dims,
            out_dims=self.decoder_out_dims,
            spatial_size=self.sim_cfg.object_grid_size,
        )
        return decoder

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.trainer.estimated_stepping_batches),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predicts whitened aberration coefficients from a [B, num_z, H, W] batch of normalized z-stacks"""
        return self.decoder(images)

    def predict_coefficients(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts physical aberration coefficients from a [B, num_z, H, W] batch of normalized z-stacks"""
        return self.unwhiten_predictions(self.predicted_means(self.forward(images)))

    def predicted_means(self, predictions: torch.Tensor) -> torch.Tensor:
        """Whitened point estimates, may be overridden by child classes that predict additional parameters"""
        return predictions

    @property
    def batch_size(self) -> int:
        """Convenience accessor because we call it often, other params should be accessed via `self.train_cfg`"""
        return self.train_cfg.batch_size

    @property
    def decoder_out_dims(self) -> int:
        """Number of decoder output dims, may be overridden by child classes that predict additional parameters"""
        return self.num_phase_coefs + self.num_amp_coefs

    # Coefficient bookkeeping

    @property
    def num_phase_coefs(self) -> int:
        """Convenience pass-through"""
        return self.propagator.num_phase_coefs

    @property
    def num_amp_coefs(self) -> int:
        """Convenience pass-through"""
        return self.propagator.num_amp_coefs

    @property
    def num_coefs(self) -> int:
        """Convenience pass-through"""
        return self.num_phase_coefs + self.num_amp_coefs

    @property
    def num_nonpiston_coefs(self) -> int:
        """Number of non-piston coefficients modeled"""
        return int((~self.is_piston).sum())

    @property
    def num_nonpiston_phase_coefs(self) -> int:
        """Number of non-piston phase coefficients, i.e. where `nonpiston_labels` switches Z -> A"""
        return int((~self.is_phase_piston).sum())

    @property
    def num_nonpiston_amp_coefs(self) -> int:
        """Number of non-piston phase coefficients"""
        return int((~self.is_amp_piston).sum())

    @property
    def coefficient_labels(self) -> list[str]:
        """Zernike coef labels for all [N_tot,] coefficients, phase then amp: [`Z_n^m`, ..., `A_n^m`, ...]"""
        return (self.propagator.phase_projector.format_labels("Z")
                + self.propagator.amp_projector.format_labels("A"))

    @property
    def nonpiston_labels(self) -> list[str]:
        """`coefficient_labels`, restricted to the [N_kept,] non-piston coefficients"""
        return [label for label, piston in zip(self.coefficient_labels, self.is_piston.tolist()) if not piston]

    def join_phase_amp(self, phase_coefs: torch.Tensor, amp_coefs: torch.Tensor) -> torch.Tensor:
        """Cats [..., num_phase] and [..., num_amp] coefs into [..., N_tot]"""
        return torch.cat([phase_coefs, amp_coefs], dim=-1)

    def split_phase_amp(self, coefs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits a [..., N_tot] tensor into its phase and amplitude parts"""
        return coefs[..., :self.num_phase_coefs], coefs[..., self.num_phase_coefs:]

    def gather_nonpiston(self, coefs: torch.Tensor) -> torch.Tensor:
        """Selects the [..., N_kept] non-piston coefficients of the full [..., N_tot]"""
        return coefs[..., ~self.is_piston]

    def scatter_nonpiston(self, kept_coefs: torch.Tensor) -> torch.Tensor:
        """
        Scatters [..., N_kept] non-piston coefficients back into a full [..., N_tot] whitened vector.
        The "whitened" part is important, since it means the piston slots can be left at zero.
        """
        coefs = kept_coefs.new_zeros(*kept_coefs.shape[:-1], self.num_coefs)
        coefs[..., ~self.is_piston] = kept_coefs
        return coefs

    def _setup_piston_flags(self) -> None:
        """
        Registers which coefficients are piston-types, which are handled differently throughout.
        Phase piston because the intensity doesn't care, intensity piston because we normalize anyway.
        """

        phase_nm = self.propagator.phase_projector.nm_indices
        is_phase_piston = (phase_nm == 0).all(dim=1)

        amp_nm = self.propagator.amp_projector.nm_indices
        is_amp_piston = (amp_nm == 0).all(dim=1)

        self.register_buffer("is_phase_piston", is_phase_piston)
        self.register_buffer("is_amp_piston", is_amp_piston)
        self.register_buffer("is_piston", torch.cat([self.is_phase_piston, self.is_amp_piston], dim=0))

    def _setup_jitter_decomposition(self) -> None:
        """
        Pre-calculates the projection of the defocus operator onto the phase Zernike
        basis so z-jitter can be represented in phase aberration space during training.
        """
        mask = self.propagator.pupil_mask
        is_phase_piston = self.is_phase_piston

        basis = self.propagator.phase_projector.zernike_bank[:, mask]  # [num_elements, num_pixels]
        if not is_phase_piston.any():  # make sure a piston column exists so the fitting works
            basis = torch.cat([basis, torch.ones_like(basis[:1])], dim=0)
        waves_per_defocus = self.propagator.axial_wavenumber[mask] / (2 * torch.pi)

        solution = torch.linalg.lstsq(basis.T, waves_per_defocus.unsqueeze(1)).solution
        coefs = solution[:self.num_phase_coefs, 0]  # drops the constant column, if added
        coefs = torch.where(is_phase_piston, 0.0, coefs)
        self.register_buffer("defocus_phase_coefs", coefs.to(self.ftype))  # [num_elements,]

    # Coefficient whitening

    def _setup_whitening(self) -> None:
        """
        Calculates affine scaling factors for the amplitude and phase coefficients that
        compensate for the spectral decay of the priors, allowing the model to learn
        targets of magnitude 1.

        Amplitude piston term has mean 1, other terms have mean 0 (see `generate_coefficients`).

        If `z_jitter` is set, widen the priors accordingly to keep the whitened targets magnitude 1.
        """

        phase_nm = self.propagator.phase_projector.nm_indices
        phase_means = torch.zeros(self.num_phase_coefs, dtype=self.ftype)
        phase_scales = self.phase_prior_cfg.coef_scales(phase_nm).to(self.ftype)
        if self.z_jitter > 0:
            z_jitter_adjusted = self.propagator.defocus_from_objective_z(self.z_jitter)
            jitter_var = z_jitter_adjusted**2 / 3  # variance for a uniform distrib over +/- z_jitter_adjusted
            phase_scales = torch.sqrt(phase_scales**2 + jitter_var * self.defocus_phase_coefs**2)

        amp_nm = self.propagator.amp_projector.nm_indices
        amp_means = self.is_amp_piston.to(self.ftype)
        amp_scales = self.amp_prior_cfg.coef_scales(amp_nm).to(self.ftype)

        self.register_buffer("target_means", torch.cat([phase_means, amp_means], dim=0))
        self.register_buffer("target_scales", torch.cat([phase_scales, amp_scales], dim=0))

    def whiten_targets(self, phase_coefs: torch.Tensor, amp_coefs: torch.Tensor) -> torch.Tensor:
        """Whitens the [B, N] labels and cats them into a [B, N_tot] regression target"""
        coefs = self.join_phase_amp(phase_coefs, amp_coefs)
        return ((coefs - self.target_means) / self.target_scales).float()

    def unwhiten_predictions(self, means: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unwhitens [..., N_tot] point estimates back into physical (phase, amp) coefs"""
        coefs = means * self.target_scales + self.target_means
        return self.split_phase_amp(coefs)

    # Synthetic data generation

    @staticmethod
    def normalize_stack(images: torch.Tensor) -> torch.Tensor:
        """Roughly normalizes a [..., H, W] image stack"""
        images = images - images.amin((-2, -1), keepdim=True)
        sums = images.sum((-2, -1), keepdim=True)
        return images / sums

    def _sample_pinned_coefficients(
            self,
            projector: ZernikeProjector,
            prior_cfg: PriorConfig,
            piston_mask: torch.Tensor,
            piston_value: float,
            batch_size: int,
            device: torch.device,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Samples [B, N] coefficients with RMS that decays with radial order `n`,
        pinning the piston term to `piston_value`.
        """
        nm_indices = projector.nm_indices
        coefs = torch.randn(batch_size, nm_indices.shape[0], generator=generator, dtype=self.ftype, device=device)
        coefs = coefs * prior_cfg.coef_scales(nm_indices).to(device)
        return torch.where(piston_mask, piston_value, coefs)

    def generate_phase_amp_coefficients(
            self,
            batch_size: int,
            device: torch.device,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample phase and amplitude coefficients, maintaining the
        invariant piston terms (phase = 0, amp = 1).

        Outputs are [B, num_phase] and [B, num_amp].
        """

        phase_coefs = self._sample_pinned_coefficients(
            self.propagator.phase_projector,
            self.phase_prior_cfg,
            self.is_phase_piston,
            piston_value=0,  # pin global phase
            batch_size=batch_size,
            device=device,
            generator=generator,
        )
        amp_coefs = self._sample_pinned_coefficients(
            self.propagator.amp_projector,
            self.amp_prior_cfg,
            self.is_amp_piston,
            piston_value=1,  # pin overall intensity (will be normalized)
            batch_size=batch_size,
            device=device,
            generator=generator,
        )
        return phase_coefs, amp_coefs

    def _simulate_stacks(
            self,
            z: torch.Tensor,
            phase_coefs: torch.Tensor,
            amp_coefs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forwards-simulates [B, num_z, H, W] z-stacks from [B, num_z] defocus values
        and [B, N] aberration coefficients. Propagation happens in chunks to bound memory use.
        """
        batch_size = z.shape[0]
        chunk_size = self.train_cfg.generator_chunk or batch_size
        image_chunks = []
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            z_chunk = z[start:stop].reshape(-1)
            phase_chunk = phase_coefs[start:stop].repeat_interleave(self.num_z, dim=0)
            amp_chunk = amp_coefs[start:stop].repeat_interleave(self.num_z, dim=0)
            psf = self.propagator(z_chunk, phase_chunk, amp_chunk)
            img = convolve_psf_with_object(self.object_distribution, psf)
            image_chunks.append(img.reshape(stop - start, self.num_z, *img.shape[-2:]))
        return torch.cat(image_chunks)

    def _sample_z_jitter(
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

    def _z_jitter_to_phase_coefs(self, z_offsets: torch.Tensor) -> torch.Tensor:
        """
        Returns the [..., num_phase] aberrations equivalent to the [...,] z-offsets (in objective-z units).

        Synthetic images will be calculated using the z-offsets but not this phase, while
        their phase coef labels will include this phase in place of the z-offsets.
        """
        defocus = self.propagator.defocus_from_objective_z(z_offsets)
        return defocus.unsqueeze(1) * self.defocus_phase_coefs

    def _batched_defocus(
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

    def _apply_noise(
            self,
            images: torch.Tensor,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Adds nosie to the image stack"""
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
            z = self._batched_defocus(phase_coefs.shape[0], offsets=offsets)
            images = self._simulate_stacks(z, phase_coefs, amp_coefs)
            if with_noise:
                images = self._apply_noise(images, generator)
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
        phase_coefs, amp_coefs = self.generate_phase_amp_coefficients(batch_size, self.device, generator)
        offsets = self._sample_z_jitter(batch_size, generator=generator)
        images = self.simulate_normalized_stacks(
            phase_coefs, amp_coefs, with_noise=True, offsets=offsets, generator=generator
        )
        phase_coefs = phase_coefs + self._z_jitter_to_phase_coefs(offsets)
        return images, phase_coefs, amp_coefs

    def _pacing_loader(self, num_batches: int) -> DataLoader:
        """Placeholder loader used only to pace the loop (actual batches are synthesized inside the steps)"""
        return DataLoader(TensorDataset(torch.arange(num_batches)), batch_size=1)

    def train_dataloader(self) -> DataLoader:
        return self._pacing_loader(self.train_cfg.steps_per_epoch)

    def val_dataloader(self) -> DataLoader:
        return self._pacing_loader(self.train_cfg.val_batches)

    # PTL training methods

    def compute_losses(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns the total loss and a dict of individual losses for whitened predictions
        vs. targets. May be overridden by child classes that predict additional parameters.
        """
        keep = ~self.is_piston  # mask piston terms out of the loss
        loss = F.mse_loss(predictions[:, keep], targets[:, keep])
        return loss, {}

    def val_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Validation-only scalars for logging"""
        _, amp = self.split_phase_amp(self.predicted_means(predictions))
        _, amp_targets = self.split_phase_amp(targets)
        return {"amp_rmse_whitened": F.mse_loss(amp, amp_targets).sqrt()}

    def _forwards_common(
            self,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Generates synthetic training pairs, runs inference from the images, and calculates losses"""
        images, phase_coefs, amp_coefs = self.create_examples(self.batch_size, generator=generator)
        targets = self.whiten_targets(phase_coefs, amp_coefs)
        predictions = self.forward(images)
        loss, logs = self.compute_losses(predictions, targets)
        return loss, predictions, targets, logs

    def _log_stage(self, stage: str, loss: torch.Tensor, logs: dict) -> None:
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=self.batch_size)
        for name, value in logs.items():
            self.log(f"{stage}/{name}", value, batch_size=self.batch_size)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, _, _, logs = self._forwards_common()
        self._log_stage("train", loss, logs)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(self.train_cfg.val_seed + batch_idx)
        loss, predictions, targets, logs = self._forwards_common(generator=generator)
        self._log_stage("val", loss, logs)
        for name, value in self.val_metrics(predictions, targets).items():
            self.log(f"val/{name}", value, batch_size=self.batch_size)
        return loss


class ZstackSolver_Heteroscedastic(ZstackSolver):
    """
    Extends ZstackSolver, training a decoder to predict not only the most likely
    aberration coefficients, but also their individual uncertainties.
    Each coefficient is modeled as its own Gaussian distribution, so correlations
    and multimodality are still not handled. Maybe someday!
    """
    @property
    def decoder_out_dims(self) -> int:
        """mu and log(sigma) for each phase and amplitude coefficient"""
        return 2 * (self.num_phase_coefs + self.num_amp_coefs)

    def _whitened_normal(self, predictions: torch.Tensor) -> torch.distributions.Normal:
        """Reads the raw decoder output as a per-coefficient Gaussian in whitened space"""
        mu, log_sigma = predictions.chunk(2, dim=1)
        return torch.distributions.Normal(mu, log_sigma.exp())

    def predicted_means(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions.chunk(2, dim=1)[0]

    def compute_losses(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        dist = self._whitened_normal(predictions)
        keep = ~self.is_piston  # mask piston terms out of the losses
        mu_loss = F.mse_loss(dist.mean[:, keep], targets[:, keep])
        sigma_loss = F.gaussian_nll_loss(
            dist.mean.detach()[:, keep], targets[:, keep], dist.variance[:, keep]
        )  # detach the mean so the sigma head can't push it back to the uninformative priors
        return mu_loss + sigma_loss, {"mu_mse": mu_loss, "sigma_nll": sigma_loss}

    def val_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        metrics = super().val_metrics(predictions, targets)
        dist = self._whitened_normal(predictions)
        keep = ~self.is_piston
        z_scores = ((dist.mean - targets) / dist.stddev)[:, keep]
        metrics["z_std"] = z_scores.std()  # ~1 when the sigmas are calibrated
        return metrics

    def predict_distribution(self, images: torch.Tensor) -> torch.distributions.Independent:
        """
        Predicts the physical-space joint coefficient distributions (excluding the pinned piston coefs).
        Joint distribution is diagonal, but wrap it as the marginal of a full pairwise multivariate
        distribution for API consistency with other solver variants.
        """
        dist = self._whitened_normal(self.forward(images))
        keep = ~self.is_piston
        scales = self.target_scales[keep]
        means = self.target_means[keep]
        marginals = torch.distributions.Normal(
            dist.mean[:, keep] * scales + means,
            dist.stddev[:, keep] * scales,
        )
        return torch.distributions.Independent(marginals, reinterpreted_batch_ndims=1)


class ZstackSolver_Covariance(ZstackSolver):
    """
    Extends ZstackSolver, training a decoder to predict not only the most likely
    aberration coefficients, but also their full covariance matrix.

    Handles correlations (at least, pairwise ones), but still not multimodality.

    The covariance matrix is represented by its Cholensky factor, a flattened
    version of which is predicted by the extra decoder head.
    """
    def _setup_whitening(self) -> None:
        """Also sets up index bookkeeping for the flattened Cholesky factor"""
        super()._setup_whitening()
        tril_indices = torch.tril_indices(self.num_nonpiston_coefs, self.num_nonpiston_coefs)
        self.register_buffer("tril_indices", tril_indices)
        self.register_buffer("tril_is_diagonal", tril_indices[0] == tril_indices[1])

    @property
    def decoder_out_dims(self) -> int:
        """mu for every coefficient, plus the flattened Cholesky factor over the non-piston ones"""
        return (self.num_phase_coefs + self.num_amp_coefs) + self.tril_indices.shape[1]

    def _setup_decoder(
            self,
            hidden_channels: Sequence[int],
            embedding_dims: int,
    ) -> nn.Module:
        """
        Sets up a decoder with an extra, detached output head for the Cholensky factor.
        Initializes with no covariance (Cholensky = identity matrix).
        """
        decoder = CnnDecoder_DetachedHead(
            in_channels=self.num_z,
            spatial_hidden_channels=hidden_channels,
            embedding_dims=embedding_dims,
            out_dims=self.num_phase_coefs + self.num_amp_coefs,
            detached_out_dims=self.tril_indices.shape[1],
            spatial_size=self.sim_cfg.object_grid_size,
        )

        inverse_softplus_one = math.log(math.expm1(1.0))  # so softplus of diagonal = 1
        nn.init.zeros_(decoder.detached_head.weight)
        with torch.no_grad():
            decoder.detached_head.bias.copy_(
                torch.where(self.tril_is_diagonal, inverse_softplus_one, 0.0)
            )
        return decoder

    def predicted_means(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions[:, :self.num_phase_coefs + self.num_amp_coefs]

    def _to_scale_tril(self, chol_entries: torch.Tensor) -> torch.Tensor:
        """
        Converts the flattened [B, K] Cholensky factors to lower-triangular [B, N_kept, N_kept].
        Also applies softplus to the diagonal, to keep the Cholensky decomposition unique.
        """
        chol_entries = torch.where(self.tril_is_diagonal, F.softplus(chol_entries) + 1e-6, chol_entries)
        scale_tril = chol_entries.new_zeros(chol_entries.shape[0], self.num_nonpiston_coefs, self.num_nonpiston_coefs)
        scale_tril[:, self.tril_indices[0], self.tril_indices[1]] = chol_entries
        return scale_tril

    def _whitened_normal(self, predictions: torch.Tensor) -> torch.distributions.MultivariateNormal:
        """Reads the raw decoder output as a joint Gaussian in whitened space"""
        means = self.predicted_means(predictions)[:, ~self.is_piston]
        scale_tril = self._to_scale_tril(predictions[:, self.num_phase_coefs + self.num_amp_coefs:])
        return torch.distributions.MultivariateNormal(means, scale_tril=scale_tril)

    def compute_losses(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        dist = self._whitened_normal(predictions)
        targets_kept = targets[:, ~self.is_piston]
        mu_loss = F.mse_loss(dist.mean, targets_kept)
        detached_dist = torch.distributions.MultivariateNormal(
            dist.mean.detach(), scale_tril=dist.scale_tril
        )  # detach the mean so the covariance head can't push it back to the uninformative priors
        sigma_loss = -detached_dist.log_prob(targets_kept).mean() / self.num_nonpiston_coefs  # per-coefficient scale
        return mu_loss + sigma_loss, {"mu_mse": mu_loss, "sigma_nll": sigma_loss}

    def val_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        metrics = super().val_metrics(predictions, targets)
        dist = self._whitened_normal(predictions)
        residuals = (targets[:, ~self.is_piston] - dist.mean).unsqueeze(-1)
        z_scores = torch.linalg.solve_triangular(dist.scale_tril, residuals, upper=False)
        metrics["z_std"] = z_scores.std()  # ~1 when the covariances are calibrated
        return metrics

    def predict_distribution(self, images: torch.Tensor) -> torch.distributions.MultivariateNormal:
        """Predicts the physical-space joint coefficient distributions (excluding the pinned piston coefs)"""
        dist = self._whitened_normal(self.forward(images))
        scales = self.target_scales[~self.is_piston]
        means = self.target_means[~self.is_piston]
        return torch.distributions.MultivariateNormal(
            dist.mean * scales + means,
            scale_tril=scales.unsqueeze(-1) * dist.scale_tril,
        )


class ZstackSolver_MixedDensity(ZstackSolver):
    """
    Extends ZstackSolver, training a decoder to predict a Gaussian mixture over the aberration coefficients.
    Finally handles multimodality!

    Models up to `num_components` components, each parameterized like the single Gaussian
    of ZstackSolver_Covariance (i.e., component mean + flattened Cholesky factor), joined
    by a mixing coefficient for each component.

    `mixing_warmup_epochs` holds the mixing weights uniform and frozen for the
    first epochs so the component means can specialize before the weights are learned.
    Without it, multimodal targets tend to collapse onto a single component early.
    """
    def __init__(
            self,
            *,
            train_cfg: TrainingConfig,
            sim_cfg: SimulationConfig,
            optics_cfg: OpticalConfig,
            phase_cfg: ZernikeConfig,
            amp_cfg: ZernikeConfig,
            phase_prior_cfg: PriorConfig,
            amp_prior_cfg: PriorConfig,
            object_distribution: torch.Tensor,
            z_objective: Optional[torch.Tensor] = None,
            z_jitter: float = 0.0,
            noise_cfg: Optional[NoiseConfig] = None,
            hidden_channels: Sequence[int] = (16, 32),
            embedding_dims: int = 128,
            num_components: int = 4,
    ):
        self.num_components = num_components  # must come before decoder init
        super().__init__(
            train_cfg=train_cfg,
            sim_cfg=sim_cfg,
            optics_cfg=optics_cfg,
            phase_cfg=phase_cfg,
            amp_cfg=amp_cfg,
            phase_prior_cfg=phase_prior_cfg,
            amp_prior_cfg=amp_prior_cfg,
            z_objective=z_objective,
            z_jitter=z_jitter,
            object_distribution=object_distribution,
            noise_cfg=noise_cfg,
            hidden_channels=hidden_channels,
            embedding_dims=embedding_dims,
        )

    def _setup_whitening(self) -> None:
        """Also sets up index bookkeeping for the flattened Cholesky factors"""
        super()._setup_whitening()
        tril_indices = torch.tril_indices(self.num_nonpiston_coefs, self.num_nonpiston_coefs)
        self.register_buffer("tril_indices", tril_indices)
        self.register_buffer("tril_is_diagonal", tril_indices[0] == tril_indices[1])

    @property
    def decoder_out_dims(self) -> int:
        """Per component: mu for every coefficient, flattened Cholesky factor, and a mixing logit"""
        return self.num_components * (self.num_coefs + self.tril_indices.shape[1] + 1)

    def _setup_decoder(
            self,
            hidden_channels: Sequence[int],
            embedding_dims: int,
    ) -> nn.Module:
        """
        Sets up a decoder with an extra, detached output head for the Cholensky factors and the mixing logits.
        Initializes with no covariance (Cholensky = identity matrix).
        """
        num_chol = self.num_components * self.tril_indices.shape[1]
        decoder = CnnDecoder_DetachedHead(
            in_channels=self.num_z,
            spatial_hidden_channels=hidden_channels,
            embedding_dims=embedding_dims,
            out_dims=self.num_components * self.num_coefs,
            detached_out_dims=self.num_components + num_chol,
            spatial_size=self.sim_cfg.object_grid_size,
        )

        inverse_softplus_one = math.log(math.expm1(1.0))  # so softplus of diagonal = 1
        nn.init.zeros_(decoder.detached_head.weight)
        with torch.no_grad():
            chol_bias = torch.where(self.tril_is_diagonal, inverse_softplus_one, 0.0)
            logit_bias = torch.zeros(self.num_components)
            decoder.detached_head.bias.copy_(
                torch.cat([logit_bias, chol_bias.repeat(self.num_components)])
            )

            generator = torch.Generator().manual_seed(0)
            mean_bias = torch.randn(self.num_components, self.num_coefs, generator=generator)
            decoder.project.bias.copy_(mean_bias.flatten())  # seed with an O(1) spread to encourage specialization
        return decoder

    def _split_predictions(
            self,
            predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method, splits the raw [B, D] decoder output into:
        - means: [B, K, N] (in whitened space)
        - logits: [B, K] mixing logits
        - chol_entries: [B, K, T] flattened Cholesky factors
        """
        num_means = self.num_components * self.num_coefs
        means = predictions[:, :num_means].reshape(-1, self.num_components, self.num_coefs)
        logits = predictions[:, num_means:num_means + self.num_components]
        chol_entries = predictions[:, num_means + self.num_components:].reshape(
            -1, self.num_components, self.tril_indices.shape[1]
        )
        return means, logits, chol_entries

    def predicted_means(self, predictions: torch.Tensor) -> torch.Tensor:
        """Whitened means of the highest-weight component"""
        means, logits, _ = self._split_predictions(predictions)
        batch_idx = torch.arange(means.shape[0], device=means.device)
        return means[batch_idx, logits.argmax(dim=1)]

    def _to_scale_tril(self, chol_entries: torch.Tensor) -> torch.Tensor:
        """
        Converts the flattened [..., T] Cholesky factors to lower-triangular [..., N_kept, N_kept].
        Also applies softplus to the diagonal, to keep the Cholesky decomposition unique.
        """
        chol_entries = torch.where(self.tril_is_diagonal, F.softplus(chol_entries) + 1e-6, chol_entries)
        scale_tril = chol_entries.new_zeros(*chol_entries.shape[:-1], self.num_nonpiston_coefs, self.num_nonpiston_coefs)
        scale_tril[..., self.tril_indices[0], self.tril_indices[1]] = chol_entries
        return scale_tril

    def _whitened_mixture(self, predictions: torch.Tensor) -> torch.distributions.MixtureSameFamily:
        """Reads the raw decoder output as a Gaussian mixture in whitened space (non-piston coefs only)"""
        means, logits, chol_entries = self._split_predictions(predictions)
        components = torch.distributions.MultivariateNormal(
            means[..., ~self.is_piston], scale_tril=self._to_scale_tril(chol_entries)
        )
        weights = torch.distributions.Categorical(logits=logits)
        return torch.distributions.MixtureSameFamily(weights, components)

    def _mixing_logits(self, mixture: torch.distributions.MixtureSameFamily) -> torch.Tensor:
        """
        Mixing logits used by the losses.

        During the mixing warmup epochs, the learned logits are replaced with uniform constants
        to keep them frozen. This lets the components specialize before the mixture weights
        are learned, so multimodal targets don't collapse too early.
        """
        logits = mixture.mixture_distribution.logits
        if self.current_epoch < self.train_cfg.mixing_warmup_epochs:
            return torch.zeros_like(logits)
        return logits

    def compute_losses(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        mixture = self._whitened_mixture(predictions)
        dist = mixture.component_distribution
        mixing_logits = self._mixing_logits(mixture)
        targets_kept = targets[:, ~self.is_piston]

        with torch.no_grad():  # allocations to components under current mixture, Prob(component | target)
            log_joint = mixing_logits + dist.log_prob(targets_kept.unsqueeze(1))
            allocations = log_joint.softmax(dim=1)

        dist_mse = (dist.mean - targets_kept.unsqueeze(1)).square().mean(dim=2)
        mu_loss = (allocations * dist_mse).sum(dim=1).mean()  # allocation-weighted MSE

        detached_mixture = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=mixing_logits),
            torch.distributions.MultivariateNormal(
                dist.mean.detach(), scale_tril=dist.scale_tril
            ),
        )  # detach the mean so the covariance head can't push it back to the uninformative priors
        sigma_nll = -detached_mixture.log_prob(targets_kept).mean() / self.num_nonpiston_coefs  # per-coefficient scale

        weight_entropy = mixture.mixture_distribution.entropy().mean()  # 0 if sticks to one component, log(K) else
        logs = {"mu_mse": mu_loss, "sigma_nll": sigma_nll, "weight_entropy": weight_entropy}
        return mu_loss + sigma_nll, logs

    def val_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        metrics = super().val_metrics(predictions, targets)
        means, logits, _ = self._split_predictions(predictions)

        max_allocations = logits.softmax(dim=1).amax(dim=1)
        metrics["dominant_component_allocation"] = max_allocations.mean()

        residuals = means[..., ~self.is_piston] - targets[:, ~self.is_piston].unsqueeze(1)
        component_mse = residuals.square().mean(dim=2)  # [B, K]
        batch_idx = torch.arange(means.shape[0], device=means.device)
        metrics["dominant_component_rmse"] = component_mse[batch_idx, logits.argmax(dim=1)].mean().sqrt()
        metrics["best_component_rmse"] = component_mse.amin(dim=1).mean().sqrt()

        return metrics

    def predict_distribution(self, images: torch.Tensor) -> torch.distributions.MixtureSameFamily:
        """Predicts the physical-space mixture distributions (excluding the pinned piston coefs)"""
        means, logits, chol_entries = self._split_predictions(self.forward(images))
        scales = self.target_scales[~self.is_piston]
        components = torch.distributions.MultivariateNormal(
            means[..., ~self.is_piston] * scales + self.target_means[~self.is_piston],
            scale_tril=scales.unsqueeze(-1) * self._to_scale_tril(chol_entries),
        )
        return torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=logits), components
        )

    def predict_component_coefficients(
            self,
            images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the physical-space joint coefficient distributions for each component.

        Returns:
        - weights: [B, K] mixture weights
        - phase_coefs: [B, K, num_phase_coefs]
        - amp_coefs: [B, K, num_amp_coefs]
        """
        means, logits, _ = self._split_predictions(self.forward(images))
        coefs = means * self.target_scales + self.target_means
        return logits.softmax(dim=1), coefs[..., :self.num_phase_coefs], coefs[..., self.num_phase_coefs:]


class ZstackSolver_FlowMatching(ZstackSolver):
    """
    Extends ZstackSolver, training a conditional velocity field that transports a standard
    normal into the posterior over aberration coefficients, given a z-stack.

    Unlike the mixture model, the posterior is represented implicitly: there is no closed-form
    density, only the ability to draw samples by integrating the learned ODE. In exchange the
    training objective is a plain regression, so none of the mixture's failure modes
    (collapsing weights, covariance-preconditioned mean gradients) apply.

    The flow lives in *whitened* space over the non-piston coefficients only, where the
    sampling prior is already close to a standard normal -- so the flow only has to sharpen
    a roughly-correct prior rather than build the distribution from nothing.
    """
    def __init__(
            self,
            *,
            train_cfg: TrainingConfig,
            sim_cfg: SimulationConfig,
            optics_cfg: OpticalConfig,
            phase_cfg: ZernikeConfig,
            amp_cfg: ZernikeConfig,
            phase_prior_cfg: PriorConfig,
            amp_prior_cfg: PriorConfig,
            vel_cfg: VelocityConfig,
            object_distribution: torch.Tensor,
            z_objective: Optional[torch.Tensor] = None,
            z_jitter: float = 0.0,
            noise_cfg: Optional[NoiseConfig] = None,
            num_val_samples: int = 64,
            ode_solver: Optional[ODESolver] = None,
            hidden_channels: Sequence[int] = (16, 32),
            embedding_dims: int = 128,
    ):
        self.vel_cfg = vel_cfg  # must come before decoder init
        super().__init__(
            train_cfg=train_cfg,
            sim_cfg=sim_cfg,
            optics_cfg=optics_cfg,
            phase_cfg=phase_cfg,
            amp_cfg=amp_cfg,
            phase_prior_cfg=phase_prior_cfg,
            amp_prior_cfg=amp_prior_cfg,
            z_objective=z_objective,
            z_jitter=z_jitter,
            object_distribution=object_distribution,
            noise_cfg=noise_cfg,
            hidden_channels=hidden_channels,
            embedding_dims=embedding_dims,
        )

        self.num_val_samples = num_val_samples
        if ode_solver is None:  # instantiate fresh, in case future solvers hold mutable state
            ode_solver = EulerSolver()
        self.ode_solver = ode_solver

    @property
    def flow_dims(self) -> int:
        """Dimension the flow transports: the non-piston coefficients, in whitened space"""
        return self.num_nonpiston_coefs

    @property
    def decoder_out_dims(self) -> int:
        raise NotImplementedError("ZstackSolver_FlowMatching does not use `decoder_out_dims`")

    def _setup_decoder(
            self,
            hidden_channels: Sequence[int],
            embedding_dims: int,
    ) -> nn.Module:
        """Sets up the conditional velocity field"""
        return VelocityField(
            in_channels=self.num_z,
            spatial_hidden_channels=hidden_channels,
            image_embedding_dims=embedding_dims,
            spatial_size=self.sim_cfg.object_grid_size,
            flow_dims=self.flow_dims,
            time_embedding_dims=self.vel_cfg.time_embedding_dims,
            hidden_dims=self.vel_cfg.hidden_dims,
            is_residual=self.vel_cfg.is_residual,
            residual_dims=self.vel_cfg.residual_dims,
        )

    # Whitened-space bookkeeping

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a [B, num_z, H, W] batch of z-stacks as [B, embedding_dims] conditioning vectors.

        Note: unlike the rest of the ZstackSolver family, this is not a coefficient estimate.
        Inferring the aberration coefficients requires flow integration (see `sample`).

        Follows the theory laid out in Lipman et al., "Flow matching for generative
        modeling," arXiv:2210.02747v2 (2023).
        """
        return self.decoder.encode(images)

    # Flow matching

    def sample_flow_pairs(
            self,
            x_1: torch.Tensor,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draws conditional-optimal-transport training pairs for a [B, N_kept] batch of whitened targets.

        Arguments:
        - x_1: [B, N_kept] flow-space targets (the data end of the path)
        Returns:
        - x_t: [B, N_kept] points along the straight path
        - t: [B,] flow times, uniform on [0, 1]
        - u_t: [B, N_kept] target velocities
        """
        batch_size = x_1.shape[0]
        kwargs = dict(generator=generator, dtype=x_1.dtype, device=x_1.device)

        x_0 = torch.randn(x_1.shape, **kwargs)  # noise end of the path
        t = torch.rand(batch_size, **kwargs)

        x_t = torch.lerp(x_0, x_1, t.unsqueeze(-1))  # (1 - t) * x_0 + t * x_1
        u_t = x_1 - x_0  # conditional-optimal-transport velocity (constant along the path)

        return x_t, t, u_t

    def flow_loss(
            self,
            y: torch.Tensor,
            targets: torch.Tensor,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculates the conditional flow-matching loss as the MSE between predicted and target velocities.
        Returns the total loss and a dict of individual losses, matching `super().compute_losses`.
        """
        x_1 = self.gather_nonpiston(targets)
        x_t, t, u_t = self.sample_flow_pairs(x_1, generator)
        v_t = self.decoder(x_t, t, y)
        cfm_loss = F.mse_loss(v_t, u_t)
        return cfm_loss, {"cfm_loss": cfm_loss}

    def compute_losses(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Unseeded passthrough to `flow_loss`, for compatibility with the base class"""
        return self.flow_loss(predictions, targets)

    def _forwards_common(
            self,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        As the base class, but threads `generator` into the loss as well as the data
        generation, since the flow objective samples x_0 and t on top of the training pair.
        The second return slot carries the conditioning vectors rather than predictions.
        """
        images, phase_coefs, amp_coefs = self.create_examples(self.batch_size, generator=generator)
        targets = self.whiten_targets(phase_coefs, amp_coefs)
        y = self.forward(images)
        loss, logs = self.flow_loss(y, targets, generator=generator)
        return loss, y, targets, logs

    # Sampling and inference

    @torch.no_grad()
    def sample(
            self,
            y: torch.Tensor,
            num_samples: int = 1,
            num_steps: Optional[int] = None,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Draws posterior samples by integrating the velocity field from t=0 to t=1.

        Arguments:
        - y: [B, embedding_dims] conditioning vectors from `forward`
        - num_samples: posterior samples to draw per conditioning vector
        - num_steps: integration steps to take, defaulting to `vel_cfg.num_sample_steps`
        Returns:
        - [B, num_samples, N_kept] whitened flow-space samples
        """
        if num_steps is None:
            num_steps = self.vel_cfg.num_sample_steps

        batch_size = y.shape[0]
        y_expanded = y.repeat_interleave(num_samples, dim=0)

        x_0 = torch.randn(
            batch_size * num_samples,
            self.flow_dims,
            generator=generator,
            dtype=y.dtype,
            device=y.device,
        )  # same prior the training pairs start from

        def velocity(x: torch.Tensor, t: float) -> torch.Tensor:
            """Expands scalar t to batch, to keep the ODESolver call clean"""
            t_batch = torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device)
            return self.decoder(x, t_batch, y_expanded)

        x_1 = self.ode_solver.integrate(velocity, x_0, num_steps)
        return x_1.reshape(batch_size, num_samples, self.flow_dims)

    def predicted_means(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Whitened point estimate, as the mean over `num_val_samples` posterior samples.
        Scattered back to full [B, N_tot] width, so the inherited `predict_coefficients` works.

        This isn't terribly meaningful if the posterior distribution is multimodal.
        Judge multimodal systems by `best_sample_rmse` instead.
        """
        samples = self.sample(predictions, num_samples=self.num_val_samples)
        return self.scatter_nonpiston(samples.mean(dim=1))

    def predict_samples(
            self,
            images: torch.Tensor,
            num_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draws physical-space posterior samples from a [B, num_z, H, W] batch of z-stacks.
        Comparable to `predict_component_coefficients` in other ZstackSolver classes.

        Returns:
        - phase_coefs: [B, num_samples, num_phase_coefs]
        - amp_coefs: [B, num_samples, num_amp_coefs]
        """
        y = self.forward(images)
        x_1 = self.sample(y, num_samples=num_samples)
        return self.unwhiten_predictions(self.scatter_nonpiston(x_1))

    def val_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Validation-only scalars for logging.

        Samples from the predicted posterior distributions, then reports
        statistics comparing those distributions to the targets.

        Also compares the truth to the sampled distribution by ordering them, then mapping
        the rankings to [0, 1]. If the truth belongs in the distrib, it's equally likely
        to be in any rank, so the distribution of rankings should be uniform. Compare the
        stdev to that of a uniform distribution to check calibration.

        Compared to other ZstackSolver classes, these metrics carry some noise from
        the random sampling. Could reseed the RNG I guess, but it's probably fine.
        """
        samples = self.sample(predictions, num_samples=self.num_val_samples)  # [B, num_samples, N_kept]
        targets_kept = self.gather_nonpiston(targets).unsqueeze(1)  # [B, 1, N_kept]

        _, amp = self.split_phase_amp(self.scatter_nonpiston(samples.mean(dim=1)))
        _, amp_targets = self.split_phase_amp(targets)
        metrics = {"amp_rmse_whitened": F.mse_loss(amp, amp_targets).sqrt()}

        sample_mse = (samples - targets_kept).square().mean(dim=2)  # [B, num_samples]
        metrics["best_sample_rmse"] = sample_mse.amin(dim=1).mean().sqrt()  # cf. best_component_rmse
        metrics["sample_mean_rmse"] = (samples.mean(dim=1) - targets_kept.squeeze(1)).square().mean().sqrt()
        metrics["sample_spread"] = samples.std(dim=1).mean()  # ~0 if flow collapsed to a deterministic map

        ranks = (samples < targets_kept).sum(dim=1) / self.num_val_samples  # [B, N_kept], in [0, 1]
        num_ranks = self.num_val_samples + 1
        uniform_std = math.sqrt((num_ranks**2 - 1) / 12) / self.num_val_samples
        metrics["rank_std"] = ranks.std() / uniform_std  # ~1 when spread is calibrated

        return metrics
