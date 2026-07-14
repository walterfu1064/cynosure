"""
PyTorch Lightning model that trains itself to infer phase and amplitude
aberrations from z-stacks.

Training data is synthesized on the fly using Zernike coefficients sampled
from a decaying-spectrum prior, and propagating forwards to get the synthetic images.

Aberration coefficients are scaled/unscaled before/after using to train,
to keep them on an easy scale for the model to learn.

Validation patches are generated in the same way, but with a fixed seed so
they're consistent across epochs.
"""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .submodules import build_encoder
from ..beam_propagation import BeamPropagator
from ..config import SimulationConfig, OpticalConfig, PriorConfig, ZernikeConfig
from ..utilities.fft_utilities import convolve_psf_with_object


def make_object_distribution(
        object_grid_size: int,
        object_pixel_size: float,
        bead_diameter: float,
) -> torch.Tensor:
    x = torch.arange(-(object_grid_size//2), object_grid_size//2 + 1) * object_pixel_size
    y = torch.arange(-(object_grid_size//2), object_grid_size//2 + 1) * object_pixel_size
    Y, X = torch.meshgrid(y, x, indexing="ij")
    R = torch.sqrt(X*X + Y*Y)
    object_distrib = (R <= bead_diameter/2).float()
    return object_distrib


class Zstack_Solver(pl.LightningModule):
    def __init__(
            self,
            sim_cfg: SimulationConfig,
            optics_cfg: OpticalConfig,
            phase_cfg: ZernikeConfig,
            amp_cfg: ZernikeConfig,
            phase_prior_cfg: PriorConfig,
            amp_prior_cfg: PriorConfig,

            z_objective: torch.Tensor,
            object_distribution: torch.Tensor,

            learning_rate: float,
            weight_decay: float,

            batch_size: int = 32,
            generator_chunk: int = 4,
            steps_per_epoch: int = 200,
            val_batches: int = 16,
            val_seed: int = 42,
    ):
        super().__init__()

        self.sim_cfg = sim_cfg
        self.optics_cfg = optics_cfg
        self.phase_cfg = phase_cfg
        self.amp_cfg = amp_cfg
        self.phase_prior_cfg = phase_prior_cfg
        self.amp_prior_cfg = amp_prior_cfg

        self.propagator = BeamPropagator(sim_cfg, optics_cfg, phase_cfg, amp_cfg=amp_cfg)
        self.ftype = self.propagator.ftype

        self.register_buffer("z_objective", z_objective.to(self.ftype))
        self.num_z = z_objective.shape[0]

        self.encoder = build_encoder(
            sim_cfg,
            phase_cfg,
            amp_cfg,
            self.num_z,
            [16, 32, 64],
            256,
        )

        self.register_buffer("object_distribution", object_distribution.to(self.ftype))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.batch_size = batch_size
        self.generator_chunk = generator_chunk
        self.steps_per_epoch = steps_per_epoch
        self.val_batches = val_batches
        self.val_seed = val_seed

        self._setup_whitening()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
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

    @property
    def num_phase_coefs(self) -> int:
        """Convenience pass-through"""
        return self.propagator.num_phase_coefs

    @property
    def num_amp_coefs(self) -> int:
        """Convenience pass-through"""
        return self.propagator.num_amp_coefs

    #%% Coefficient whitening

    @staticmethod
    def _get_coef_scales(nm_indices: torch.Tensor, noise_cfg: PriorConfig) -> torch.Tensor:
        """Returns the per-coefficient rms implied by the given prior"""
        n = nm_indices[:, 0]
        return noise_cfg.base_rms / torch.pow(n + 1, noise_cfg.decay_order)

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
        phase_scales = self._get_coef_scales(phase_nm, self.phase_prior_cfg).to(self.ftype)

        amp_nm = self.propagator.amp_projector.nm_indices
        is_amp_piston = (amp_nm == 0).all(dim=1)
        amp_means = torch.where(is_amp_piston, 1, 0).to(self.ftype)
        amp_scales = self._get_coef_scales(amp_nm, self.amp_prior_cfg).to(self.ftype)

        target_means = torch.cat([phase_means, amp_means], dim=0)
        self.register_buffer("target_means", target_means)

        target_scales = torch.cat([phase_scales, amp_scales], dim=0)
        self.register_buffer("target_scales", target_scales)

        self.register_buffer("is_phase_piston", is_phase_piston)
        self.register_buffer("is_amp_piston", is_amp_piston)

    def whiten_targets(self, phase_coefs: torch.Tensor, amp_coefs: torch.Tensor) -> torch.Tensor:
        """Whitens the [B, N] labels and cats them into a [B, N_tot] regression target"""
        coefs = torch.cat([phase_coefs, amp_coefs], dim=1)
        return (coefs - self.target_means) / self.target_scales

    def unwhiten_targets(self, predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits whitened [B, N_tot] predictions back into physical (phase, amp) coefs"""
        coefs = predictions * self.target_scales + self.target_means
        phase_coefs = coefs[:, :self.num_phase_coefs]
        amp_coefs = coefs[:, self.num_phase_coefs:]
        return phase_coefs, amp_coefs

    #%% Synthetic data generation

    @staticmethod
    def normalize_stack(images: torch.Tensor) -> torch.Tensor:
        """Roughly normalizes a [..., H, W] image stack"""
        images = images - images.amin((-2, -1), keepdim=True)
        sums = images.sum((-2, -1), keepdim=True)
        return images / sums

    @staticmethod
    def sample_coefficients(
            nm_indices: torch.Tensor,
            prior_cfg: PriorConfig,
            batch_size: int,
            *,
            generator: Optional[torch.Generator] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Samples zero-mean coefficients with RMS that decays with raidal order `n`"""
        coefs = torch.randn(batch_size, nm_indices.shape[0], generator=generator, dtype=dtype, device=device)
        n = nm_indices[:, 0].to(coefs.device)
        coefs = coefs * prior_cfg.base_rms / torch.pow(n + 1, prior_cfg.decay_order)
        return coefs

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

        phase_coefs = self.sample_coefficients(
            self.propagator.phase_projector.nm_indices,
            self.phase_prior_cfg,
            batch_size,
            generator=generator,
            dtype=self.ftype,
            device=device,
        )
        phase_coefs = torch.where(self.is_phase_piston, 0, phase_coefs)  # pin global phase

        amp_coefs = self.sample_coefficients(
            self.propagator.amp_projector.nm_indices,
            self.amp_prior_cfg,
            batch_size,
            generator=generator,
            dtype=self.ftype,
            device=device,
        )
        amp_coefs = torch.where(self.is_amp_piston, 1, amp_coefs)  # pin overall intensity (will be normalized)

        return phase_coefs, amp_coefs

    def create_examples(
            self,
            batch_size: int,
            *,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Samples a set of aberrations, forwards-simulates the z-stacks, and
        normalizes them for use as inputs to the CNN.

        Returns:
        - images: [B, num_z, H, W] normalized z-stacks
        - phase_coefs: [B, num_phase_coefs] phase aberratoin coefficients
        - amp_coefs: [B, num_amp_coefs] amp aberration coefficients
        """

        num_z = self.num_z
        device = self.z_objective.device  # TODO - make this cleaner

        with torch.no_grad():
            z = self.z_objective.unsqueeze(0).expand(batch_size, num_z)
            z = self.propagator.defocus_from_objective_z(z)
            phase_coefs, amp_coefs = self.generate_phase_amp_coefficients(batch_size, device, generator)

            # Propagation in chunks to bound memory use
            chunk_size = self.generator_chunk or batch_size
            image_chunks = []
            for start in range(0, batch_size, chunk_size):
                stop = min(start + chunk_size, batch_size)
                z_chunk = z[start:stop].reshape(-1)
                phase_chunk = phase_coefs[start:stop].repeat_interleave(self.num_z, dim=0)
                amp_chunk = amp_coefs[start:stop].repeat_interleave(self.num_z, dim=0)
                psf = self.propagator(z_chunk, phase_chunk, amp_chunk)
                img = convolve_psf_with_object(self.object_distribution, psf)
                img = img.reshape(stop - start, self.num_z, *img.shape[-2:])
                image_chunks.append(img)
            images = torch.cat(image_chunks)
            images = self.normalize_stack(images).float()

        return images, phase_coefs, amp_coefs

    def train_dataloader(self) -> DataLoader:
        """Helper method used only to pace the loop (actual batches are synthesized inside training_step)"""
        return DataLoader(TensorDataset(torch.arange(self.steps_per_epoch)), batch_size=1)

    def val_dataloader(self) -> DataLoader:
        """Helper method used only to pace the loop (actual batches are synthesized inside validation_step)"""
        return DataLoader(TensorDataset(torch.arange(self.val_batches)), batch_size=1)

    #%% PTL training methods

    def _forwards_common(
            self,
            generator: Optional[torch.Generator] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, phase_coefs, amp_coefs = self.create_examples(self.batch_size, generator=generator)
        targets = self.whiten_targets(phase_coefs, amp_coefs)
        predictions = self.encoder(images)
        loss = F.mse_loss(predictions, targets)
        return loss, predictions, targets

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._forwards_common()
        self.log("train/loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(self.val_seed + batch_idx)
        loss, predictions, targets = self._forwards_common(generator=generator)

        num_phase = self.num_phase_coefs
        self.log("val/loss", loss, prog_bar=True, batch_size=self.batch_size)
        amp_rmse = F.mse_loss(predictions[:, num_phase:], targets[:, num_phase:]).sqrt()
        self.log("val/amp_rmse_whitened", amp_rmse, batch_size=self.batch_size)
        return loss
