"""
PyTorch Lightning model that trains itself to infer phase and amplitude
aberrations from z-stacks.

Training data is synthesized on the fly by a StackSimulator, using Zernike coefficients
sampled from a decaying-spectrum prior and propagated forwards to get the synthetic images.

The ZstackSolver itself only orchestrates between the CoefficientSpace, the StackSimulator,
one of several types of PosteriorHeads, and Lightning's training machinery. The StackSimulator
handles the optics, and the PosteriorHead handles the learned network and how its outputs
are interpreted as useful quantities.

Different ZstackSolver subclasses exist solely as thin wrappers around different
types of PosteriorHeads.

Aberration coefficients are whitened/unwhitened before/after use as regression targets,
to keep them on an easy scale for the model to learn.

Validation batches are generated in the same way as training ones, but with a fixed seed
so they're consistent across epochs.
"""

from collections.abc import Sequence
from functools import partial
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .coefficients import CoefficientSpace
from .flow_matching_head import FlowMatchingHead
from .noise_model import NoiseModel
from .posterior_heads import (
    CovarianceHead,
    EncoderSpec,
    HeadFactory,
    HeteroscedasticHead,
    MixtureHead,
    PointHead,
    PosteriorHead,
)
from .stack_simulator import StackSimulator
from ..beam_propagation import BeamPropagator
from ..config import (
    MixtureConfig,
    NoiseConfig,
    OpticalConfig,
    PriorConfig,
    SimulationConfig,
    TrainingConfig,
    VelocityConfig,
    ZernikeConfig,
)
from ..object_distribution import ObjectDistribution
from ..utilities.ode_solvers import ODESolver


class ZstackSolver(pl.LightningModule):
    """
    Base for the solver family.
    Not usable on its own, just supplies shared machinery to subclasses that specify a PosteriorHead.

    Encapsulates an imaging system configuration, a z-stack decoder, and the training machinery
    that fits the latter on synthetic data generated from the former.

    Different heads express different kinds of posteriors, and should be queried different ways:
    -----------------------------------------------------------------------------------
    |   head type     | predict_coefficients | predict_distribution | predict_samples |
    |-----------------|----------------------|----------------------|-----------------|
    |      MLE        |        yes           |       raises         |     raise       |
    | Heteroscedastic |        yes           |    `Independent`     |      yes        |
    |   Covariance    |        yes           | `MultivariateNormal` |      yes        |
    |  MixedDensity   |        yes           | `MixtureSameFamily`  |      yes        |
    |  FlowMatching   |        yes           |       raises         |  yes, via ODE   |
    -----------------------------------------------------------------------------------

    `ZstackSolver_MixedDensity` also adds `predict_component_coefficients`.
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
            object_distribution: ObjectDistribution,
            z_objective: Optional[torch.Tensor] = None,
            z_jitter: float = 0.0,
            noise_cfg: Optional[NoiseConfig] = None,
            hidden_channels: Sequence[int] = (16, 32),
            embedding_dims: int = 128,
            head_factory: HeadFactory,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("object_distribution", "head_factory"))

        self.train_cfg = train_cfg
        self.sim_cfg = sim_cfg
        self.optics_cfg = optics_cfg
        self.phase_cfg = phase_cfg
        self.amp_cfg = amp_cfg
        self.phase_prior_cfg = phase_prior_cfg
        self.amp_prior_cfg = amp_prior_cfg
        self.noise_cfg = noise_cfg

        self.simulator = StackSimulator.from_configs(
            sim_cfg=sim_cfg,
            optics_cfg=optics_cfg,
            phase_cfg=phase_cfg,
            amp_cfg=amp_cfg,
            phase_prior_cfg=phase_prior_cfg,
            amp_prior_cfg=amp_prior_cfg,
            object_distribution=object_distribution,
            z_objective=z_objective,
            z_jitter=z_jitter,
            noise_cfg=noise_cfg,
            chunk_size=train_cfg.generator_chunk,
        )
        self.ftype = self.simulator.ftype
        self.z_jitter = z_jitter

        encoder = EncoderSpec(
            in_channels=self.simulator.num_z,
            spatial_hidden_channels=hidden_channels,
            embedding_dims=embedding_dims,
            spatial_size=sim_cfg.object_grid_size,
        )
        self.head: PosteriorHead = head_factory(self.simulator.coefficients, encoder)

    # Convenience accessors

    @property
    def propagator(self) -> BeamPropagator:
        return self.simulator.propagator

    @property
    def object_distribution(self) -> ObjectDistribution:
        return self.simulator.object_distribution

    @property
    def coefficients(self) -> CoefficientSpace:
        return self.simulator.coefficients

    @property
    def noise_model(self) -> Optional[NoiseModel]:
        return self.simulator.noise_model

    @property
    def num_z(self) -> int:
        return self.simulator.num_z

    @property
    def z_objective(self) -> torch.Tensor:
        return self.simulator.z_objective

    @property
    def defocus_phase_coefs(self) -> torch.Tensor:
        return self.simulator.defocus_phase_coefs

    @property
    def decoder(self) -> nn.Module:
        return self.head.decoder

    def configure_optimizers(self):
        param_groups = [{
            "params": list(self.head.parameters()),
            "weight_decay": self.train_cfg.weight_decay,
        }]

        object_params = [p for p in self.object_distribution.parameters() if p.requires_grad]
        if object_params:  # if the object is fittable, weight decay would push it to gray, so set wd=0
            param_groups.append({"params": object_params, "weight_decay": 0.0})

        optimizer = torch.optim.AdamW(param_groups, lr=self.train_cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.trainer.estimated_stepping_batches),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # Inference

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Runs the head's network on a [B, num_z, H, W] batch of normalized z-stacks.

        The results represent different things, depending on the head type.
        ZstackSolver should call `predict_coefficients` or `predict_distribution`
        instead of trying to interpret the outputs of `forward` directly.
        """
        return self.head(images)

    def predict_coefficients(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Predicts the most likely physical aberration coefficients from normalized z-stacks.
        Supported by every subclass.

        Arguments:
        - images: [B, Z, H, W] batch of normalized z-stacks (singlet dimensions as needed)
        Returns:
        - one [B, N_coefs] tensor per CoefficientBlock
        """
        return self.coefficients.unwhiten_to_blocks(self.head.whitened_means(self.forward(images)))

    def predict_distribution(self, images: torch.Tensor) -> torch.distributions.Distribution:
        """
        Predicts the physical-space coefficient distribution over the non-pinned coefs.
        Only supported by certain subclasses.

        Arguments:
        - images: [B, Z, H, W] batch of normalized z-stacks
        Returns:
        - a Distribution, type depends on head type (see class docstring)
        """
        return self.head.distribution(self.forward(images))

    def predict_samples(self, images: torch.Tensor, num_samples: int) -> tuple[torch.Tensor, ...]:
        """
        Draws physical-space posterior samples from normalized z-stacks.
        Only supported by certain subclasses.

        Arguments:
        - images: [B, Z, H, W] batch of normalized z-stacks
        - num_samples: posterior samples to draw per z-stack
        Returns:
        - one [B, num_samples, N_coefs] tensor per CoefficientBlock
        """
        whitened = self.head.whitened_samples(self.forward(images), num_samples)
        return self.coefficients.unwhiten_to_blocks(self.coefficients.scatter_nonpinned(whitened))

    # PTL training methods

    def compute_losses(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Returns the total loss and a dict of individual losses, for whitened predictions vs. targets"""
        return self.head.losses(predictions, targets, epoch=self.current_epoch, generator=generator)

    def val_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Validation-only scalars for logging"""
        return self.head.val_metrics(predictions, targets)

    def _pacing_loader(self, num_batches: int) -> DataLoader:
        """Placeholder loader used only to pace the loop (actual batches are synthesized inside the steps)"""
        return DataLoader(TensorDataset(torch.arange(num_batches)), batch_size=1)

    def train_dataloader(self) -> DataLoader:
        return self._pacing_loader(self.train_cfg.steps_per_epoch)

    def val_dataloader(self) -> DataLoader:
        return self._pacing_loader(self.train_cfg.val_batches)

    def _forwards_common(
            self,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Shared step for training.
        Generates synthetic training pairs, runs inference from the images, and calculates losses.
        """
        images, phase_coefs, amp_coefs = self.simulator.create_examples(self.train_cfg.batch_size, generator=generator)
        targets = self.coefficients.whiten_blocks(phase_coefs, amp_coefs)
        predictions = self.forward(images)
        loss, logs = self.compute_losses(predictions, targets, generator=generator)
        return loss, predictions, targets, logs

    def _log_stage(self, stage: str, loss: torch.Tensor, logs: dict) -> None:
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=self.train_cfg.batch_size)
        for name, value in logs.items():
            self.log(f"{stage}/{name}", value, batch_size=self.train_cfg.batch_size)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, _, _, logs = self._forwards_common()
        self._log_stage("train", loss, logs)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(self.train_cfg.val_seed + batch_idx)
        loss, predictions, targets, logs = self._forwards_common(generator=generator)
        self._log_stage("val", loss, logs)
        for name, value in self.val_metrics(predictions, targets).items():
            self.log(f"val/{name}", value, batch_size=self.train_cfg.batch_size)
        return loss


class ZstackSolver_MLE(ZstackSolver):
    """
    Predicts a single most-likely value per aberration coefficient, with no uncertainty.
    """
    def __init__(self, **kwargs):
        super().__init__(head_factory=PointHead, **kwargs)


class ZstackSolver_Heteroscedastic(ZstackSolver):
    """
    Predicts not only the most likely aberration coefficients, but also their individual
    uncertainties. Each coefficient is modeled as its own Gaussian, so correlations and
    multimodality are still not handled.
    """
    def __init__(self, **kwargs):
        super().__init__(head_factory=HeteroscedasticHead, **kwargs)


class ZstackSolver_Covariance(ZstackSolver):
    """
    Predicts the aberration coefficients and their full covariance matrix.
    Handles correlations (at least, pairwise ones), but still not multimodality.
    """
    def __init__(self, **kwargs):
        super().__init__(head_factory=CovarianceHead, **kwargs)


class ZstackSolver_MixedDensity(ZstackSolver):
    """
    Predicts a Gaussian mixture over the aberration coefficients,
    so multimodal posteriors can be represented.
    """

    def __init__(self, *, mixture_cfg: MixtureConfig, **kwargs):
        super().__init__(head_factory=partial(MixtureHead, cfg=mixture_cfg), **kwargs)

    @property
    def num_components(self) -> int:
        return self.head.num_components

    def predict_component_coefficients(
            self,
            images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the physical-space coefficients of each mixture component.

        Returns:
        - weights: [B, K] mixture weights
        - phase_coefs: [B, K, num_phase_coefs]
        - amp_coefs: [B, K, num_amp_coefs]
        """
        means, logits, _ = self.head.split_predictions(self.forward(images))
        phase_coefs, amp_coefs = self.coefficients.unwhiten_to_blocks(means)
        return logits.softmax(dim=1), phase_coefs, amp_coefs


class ZstackSolver_FlowMatching(ZstackSolver):
    """
    Trains a conditional velocity field that transports a standard normal into the posterior
    over the aberration coefficients, so multimodal posteriors can be represented.

    Posterior distribution is not solved in closed-form. Instead, `predict_samples` draws
    from the distribution and then integrates it out to a sample from the posterior.
    During validation, sampling is done over `num_val_samples` such samplings.

    FlowMatchingHead constructor needs a VelocityConfig and ODESolver, so pass them
    bound to a partial function for `head_factory`.
    """

    def __init__(
            self,
            *,
            vel_cfg: VelocityConfig,
            num_val_samples: int = 64,
            ode_solver: Optional[ODESolver] = None,
            **kwargs,
    ):
        super().__init__(
            head_factory=partial(
                FlowMatchingHead,
                cfg=vel_cfg,
                num_val_samples=num_val_samples,
                ode_solver=ode_solver,
            ),
            **kwargs,
        )
