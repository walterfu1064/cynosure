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

from collections.abc import Sequence
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .noise_model import NoiseModel
from .zstack_decoder import ZstackDecoder
from ..beam_propagation import BeamPropagator
from ..config import SimulationConfig, OpticalConfig, PriorConfig, ZernikeConfig, NoiseConfig
from ..utilities.fft_utilities import convolve_psf_with_object
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
            sim_cfg: SimulationConfig,
            optics_cfg: OpticalConfig,
            phase_cfg: ZernikeConfig,
            amp_cfg: ZernikeConfig,
            phase_prior_cfg: PriorConfig,
            amp_prior_cfg: PriorConfig,
            z_objective: torch.Tensor,
            object_distribution: torch.Tensor,
            noise_cfg: Optional[NoiseConfig] = None,
            learning_rate: float,
            weight_decay: float,
            hidden_channels: Sequence[int] = (16, 32),
            embedding_dims: int = 128,
            batch_size: int = 32,
            generator_chunk: int = 4,
            steps_per_epoch: int = 200,
            val_batches: int = 32,
            val_seed: int = 42,
    ):
        super().__init__()

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

        self.register_buffer("z_objective", z_objective.to(self.ftype))
        self.num_z = z_objective.shape[0]

        self.decoder = self._setup_decoder(hidden_channels, embedding_dims)

        self.register_buffer("object_distribution", object_distribution.to(self.ftype))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.batch_size = batch_size
        self.generator_chunk = generator_chunk
        self.steps_per_epoch = steps_per_epoch
        self.val_batches = val_batches
        self.val_seed = val_seed

        self._setup_whitening()

    @property
    def decoder_out_dims(self) -> int:
        """Number of decoder output dims, may be overridden by child classes that predict additional parameters"""
        return self.num_phase_coefs + self.num_amp_coefs

    def _setup_decoder(
            self,
            hidden_channels: Sequence[int],
            embedding_dims: int,
    ) -> nn.Module:
        decoder = ZstackDecoder(
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
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
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
    def num_phase_coefs(self) -> int:
        """Convenience pass-through"""
        return self.propagator.num_phase_coefs

    @property
    def num_amp_coefs(self) -> int:
        """Convenience pass-through"""
        return self.propagator.num_amp_coefs

    # Coefficient whitening

    def _setup_whitening(self) -> None:
        """
        Calculates affine scaling factors for the amplitude and phase coefficients that
        compensate for the spectral decay of the priors, allowing the model to learn
        targets of magnitude 1.

        Amplitude piston term has mean 1, other terms have mean 0 (see `generate_coefficients`).

        Also create masks for the amplitude and phase piston terms so they won't be fitted.
        Phase piston because the intensity doesn't care, intensity piston because we normalize anyway.
        """

        phase_nm = self.propagator.phase_projector.nm_indices
        is_phase_piston = (phase_nm == 0).all(dim=1)
        phase_means = torch.zeros(self.num_phase_coefs, dtype=self.ftype)
        phase_scales = self.phase_prior_cfg.coef_scales(phase_nm).to(self.ftype)

        amp_nm = self.propagator.amp_projector.nm_indices
        is_amp_piston = (amp_nm == 0).all(dim=1)
        amp_means = is_amp_piston.to(self.ftype)
        amp_scales = self.amp_prior_cfg.coef_scales(amp_nm).to(self.ftype)

        self.register_buffer("target_means", torch.cat([phase_means, amp_means], dim=0))
        self.register_buffer("target_scales", torch.cat([phase_scales, amp_scales], dim=0))
        self.register_buffer("is_phase_piston", is_phase_piston)
        self.register_buffer("is_amp_piston", is_amp_piston)
        self.register_buffer("is_piston", torch.cat([self.is_phase_piston, self.is_amp_piston], dim=0))

    def whiten_targets(self, phase_coefs: torch.Tensor, amp_coefs: torch.Tensor) -> torch.Tensor:
        """Whitens the [B, N] labels and cats them into a [B, N_tot] regression target"""
        coefs = torch.cat([phase_coefs, amp_coefs], dim=1)
        return ((coefs - self.target_means) / self.target_scales).float()

    def split_phase_amp(self, coefs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits a [B, num_phase+num_amp] tensor into its phase and amplitude parts"""
        return coefs[:, :self.num_phase_coefs], coefs[:, self.num_phase_coefs:]

    def unwhiten_predictions(self, means: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unwhitens [B, N_tot] point estimates back into physical (phase, amp) coefs"""
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
        chunk_size = self.generator_chunk or batch_size
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

    def _batched_defocus(self, batch_size: int) -> torch.Tensor:
        """Returns the [B, num_z] in-medium defocus corresponding to the stored z positions"""
        z = self.z_objective.unsqueeze(0).expand(batch_size, self.num_z)
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
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Forwards-simulates the normalized z-stacks at the model's own z positions
        from physical [B, N] coefficients (e.g. as returned by `predict_coefficients`).
        If `with_noise`, adds noise before normalizing.

        Returns [B, num_z, H, W].
        """
        with torch.no_grad():
            z = self._batched_defocus(phase_coefs.shape[0])
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

        Returns:
        - images: [B, num_z, H, W] normalized z-stacks
        - phase_coefs: [B, num_phase_coefs] phase aberratoin coefficients
        - amp_coefs: [B, num_amp_coefs] amp aberration coefficients
        """
        phase_coefs, amp_coefs = self.generate_phase_amp_coefficients(batch_size, self.device, generator)
        images = self.simulate_normalized_stacks(phase_coefs, amp_coefs, with_noise=True, generator=generator)
        return images, phase_coefs, amp_coefs

    def _pacing_loader(self, num_batches: int) -> DataLoader:
        """Placeholder loader used only to pace the loop (actual batches are synthesized inside the steps)"""
        return DataLoader(TensorDataset(torch.arange(num_batches)), batch_size=1)

    def train_dataloader(self) -> DataLoader:
        return self._pacing_loader(self.steps_per_epoch)

    def val_dataloader(self) -> DataLoader:
        return self._pacing_loader(self.val_batches)

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
        generator = torch.Generator(device=self.device).manual_seed(self.val_seed + batch_idx)
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

    def predict_distribution(self, images: torch.Tensor) -> torch.distributions.Normal:
        """Predicts the physical-space coefficient distributions"""
        dist = self._whitened_normal(self.forward(images))
        return torch.distributions.Normal(
            dist.mean * self.target_scales + self.target_means,
            dist.stddev * self.target_scales,
        )
