"""
Trains a conditional velocity field that transports a standard normal into the posterior
over aberration coefficients, given a z-stack.

Unlike the mixture model, the posterior is represented implicitly: there is no closed-form
density, only the ability to draw samples by integrating the learned ODE. In exchange the
training objective is a plain regression, so none of the mixture's failure modes
(collapsing weights, covariance-preconditioned mean gradients) apply.

The flow lives in whitened space over the non-piston coefficients, where the
sampling prior is already close to a standard normal, so the flow only has to sharpen
a roughly-correct prior rather than build the distribution from nothing.

Kept in its own module because it is the only head that needs a velocity field and an ODE
solver, and because its network output is an image embedding rather than posterior parameters.

Follows the theory laid out in Lipman et al., "Flow matching for generative modeling,"
arXiv:2210.02747v2 (2023).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coefficients import CoefficientSpace
from .posterior_heads import EncoderSpec, PosteriorHead
from .submodules import VelocityField
from ..config import VelocityConfig
from ..utilities.ode_solvers import EulerSolver, ODESolver


class FlowMatchingHead(PosteriorHead):
    """
    Posterior represented implicitly, by a conditional velocity field.

    Since there is no closed-form density, `whitened_distribution` and `distribution` both keep
    the base class's raise. `whitened_samples` is instead overridden to integrate the flow, so
    this is the one head that can be sampled from but not evaluated.

    `num_val_samples` is the draws the point estimate and the validation metrics average over.
    """

    def __init__(
            self,
            coefficients: CoefficientSpace,
            encoder: EncoderSpec,
            *,
            cfg: VelocityConfig,
            num_val_samples: int = 64,
            ode_solver: Optional[ODESolver] = None,
    ):
        self.vel_cfg = cfg
        self.num_val_samples = num_val_samples
        if ode_solver is None:  # instantiate fresh, in case future solvers hold mutable state
            ode_solver = EulerSolver()
        self.ode_solver = ode_solver
        super().__init__(coefficients, encoder)

    @property
    def flow_dims(self) -> int:
        """Dimension of the flow (i.e. the non-pinned coefficients in whitened space)"""
        return self._num_kept

    def build_network(self, encoder: EncoderSpec) -> nn.Module:
        """Sets up the conditional velocity field"""
        return VelocityField(
            in_channels=encoder.in_channels,
            spatial_hidden_channels=encoder.spatial_hidden_channels,
            image_embedding_dims=encoder.embedding_dims,
            spatial_size=encoder.spatial_size,
            flow_dims=self.flow_dims,
            time_embedding_dims=self.vel_cfg.time_embedding_dims,
            hidden_dims=self.vel_cfg.hidden_dims,
            is_residual=self.vel_cfg.is_residual,
            residual_dims=self.vel_cfg.residual_dims,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a [B, num_z, H, W] batch of z-stacks as [B, embedding_dims] conditioning vectors.

        Note: unlike the other PosterHeads, this is not a coefficient estimate.
        Inferring the aberration coefficients requires flow integration (see `whitened_samples`).
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

    def losses(
            self,
            encoded: torch.Tensor,
            targets: torch.Tensor,
            *,
            epoch: int = 0,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculates the conditional flow-matching loss as the MSE between predicted and target velocities.

        Loss calculation requires sampling, hence why `PosteriorHead.losses()` takes `generator`.
        """
        x_1 = self.coefficients.gather_nonpinned(targets)
        x_t, t, u_t = self.sample_flow_pairs(x_1, generator)
        v_t = self.decoder(x_t, t, encoded)
        cfm_loss = F.mse_loss(v_t, u_t)
        return cfm_loss, {"cfm_loss": cfm_loss}

    # Sampling and inference

    @torch.no_grad()
    def whitened_samples(
            self,
            encoded: torch.Tensor,
            num_samples: int,
            *,
            num_steps: Optional[int] = None,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Draws posterior samples by integrating the velocity field from t=0 to t=1.

        Overrides the base implementation, which draws from a closed-form density this head
        doesn't have. The extra arguments are specific to the integration, so they're
        keyword-only and default to leave the inherited two-argument call working.

        Arguments:
        - encoded: [B, embedding_dims] conditioning vectors from `forward`
        - num_samples: posterior samples to draw per conditioning vector
        - num_steps: integration steps to take, defaulting to `vel_cfg.num_sample_steps`
        Returns:
        - [B, num_samples, N_kept] whitened flow-space samples
        """
        if num_steps is None:
            num_steps = self.vel_cfg.num_sample_steps

        batch_size = encoded.shape[0]
        y_expanded = encoded.repeat_interleave(num_samples, dim=0)

        x_0 = torch.randn(
            batch_size * num_samples,
            self.flow_dims,
            generator=generator,
            dtype=encoded.dtype,
            device=encoded.device,
        )  # same prior the training pairs start from

        def velocity(x: torch.Tensor, t: float) -> torch.Tensor:
            """Expands scalar t to batch, to keep the ODESolver call clean"""
            t_batch = torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device)
            return self.decoder(x, t_batch, y_expanded)

        x_1 = self.ode_solver.integrate(velocity, x_0, num_steps)
        return x_1.reshape(batch_size, num_samples, self.flow_dims)

    def whitened_means(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Whitened point estimate, as the mean over `num_val_samples` posterior samples.
        Scattered back to full [B, N_tot] width, so the inherited `predict_coefficients` works.

        Costs a full ODE integration, unlike the analytic heads. This isn't terribly meaningful
        if the posterior is multimodal -- judge multimodal systems by `best_sample_rmse` instead.
        """
        samples = self.whitened_samples(encoded, self.num_val_samples)
        return self.coefficients.scatter_nonpinned(samples.mean(dim=1))

    def val_metrics(self, encoded: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Validation-only scalars for logging.

        Samples from the predicted posterior distributions, then reports
        statistics comparing those distributions to the targets.

        Also compares the truth to the sampled distribution by ordering them, then mapping
        the rankings to [0, 1]. If the truth belongs in the distrib, it's equally likely
        to be in any rank, so the distribution of rankings should be uniform. Compare the
        stdev to that of a uniform distribution to check calibration. This is a quick-and-dirty
        version of SBC. See `calibration.py` for a more thorough assessment.

        Compared to other PosteriorHeads, these metrics carry some noise from the random sampling.
        """
        samples = self.whitened_samples(encoded, self.num_val_samples)  # [B, num_samples, N_kept]
        targets_kept = self.coefficients.gather_nonpinned(targets).unsqueeze(1)  # [B, 1, N_kept]

        metrics = {}
        sample_means = self.coefficients.scatter_nonpinned(samples.mean(dim=1))
        if self.coefficients.has_block("object"):
            obj = self.coefficients.block_coefs(sample_means, "object")
            obj_targets = self.coefficients.block_coefs(targets, "object")
            metrics["object_rmse_whitened"] = F.mse_loss(obj, obj_targets).sqrt()

        sample_mse = (samples - targets_kept).square().mean(dim=2)  # [B, num_samples]
        metrics["best_sample_rmse"] = sample_mse.amin(dim=1).mean().sqrt()  # cf. best_component_rmse
        metrics["sample_mean_rmse"] = (samples.mean(dim=1) - targets_kept.squeeze(1)).square().mean().sqrt()
        metrics["sample_spread"] = samples.std(dim=1).mean()  # ~0 if flow collapsed to a deterministic map

        ranks = (samples < targets_kept).sum(dim=1) / self.num_val_samples  # [B, N_kept], in [0, 1]
        num_ranks = self.num_val_samples + 1
        uniform_std = math.sqrt((num_ranks**2 - 1) / 12) / self.num_val_samples
        metrics["rank_std"] = ranks.std() / uniform_std  # ~1 when spread is calibrated

        return metrics
