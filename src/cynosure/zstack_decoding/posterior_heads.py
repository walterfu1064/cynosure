"""
Output heads that predict the coefficient posteriors in various ways.

Each PosteriorHead reads the latent vector produced by an image trunk (built from an
EncoderSpec), sizes its own output projections over it, decides how to interpret those
projections' outputs, and handles its own loss calculations and validation metrics.

By and large, the ZstackSolver should only worry about a handful of call sites:
- `losses` -> called during training
- `whitened_means` -> called during training, or during inference if MLE
- `distribution` -> called during inference (if head type predicts a closed-form posterior)
- `whitened_samples` -> called during inference (if head type can be sampled from)

`FlowMatchingHead` lives in its own module, since it needs a velocity field and an
ODE solver that no other head has any use for.

Bit of hackiness: each PosteriorHead holds a reference to the CoefficientSpace
it predicts over, but deliberately doesn't register it as a child module, since
the StackSimulator owns it and will already include it in the state dict.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coefficients import CoefficientSpace
from .submodules import SetEncoder, ZstackCnnEncoder
from ..config import MixtureConfig


@dataclass(frozen=True, slots=True)
class CnnEncoderSpec:
    """Architecture of a fixed-geometry CNN image trunk for a PosteriorHead"""
    spatial_hidden_channels: Sequence[int]
    embedding_dims: int

    def build(self, *, spatial_size: int, num_z: Optional[int]) -> nn.Module:
        if num_z is None:
            raise ValueError("CnnEncoderSpec requires a fixed stack geometry")
        return ZstackCnnEncoder(
            in_channels=num_z,
            spatial_hidden_channels=self.spatial_hidden_channels,
            embedding_dims=self.embedding_dims,
            spatial_size=spatial_size,
        )


@dataclass(frozen=True, slots=True)
class SetEncoderSpec:
    """Architecture of a dynamic-z, transformer-based image-set trunk for a PosteriorHead"""
    spatial_hidden_channels: Sequence[int]
    image_embedding_dims: int
    z_embedding_dims: int
    num_attention_layers: int
    num_attention_heads: int
    attention_feedforward_dims: int
    z_min_frequency: float
    z_max_frequency: float

    def build(self, *, spatial_size: int, num_z: Optional[int] = None) -> nn.Module:
        """num_z is unused, set encoder trunk takes dynamic per-call geometry"""
        return SetEncoder(
            spatial_hidden_channels=self.spatial_hidden_channels,
            image_embedding_dims=self.image_embedding_dims,
            z_embedding_dims=self.z_embedding_dims,
            num_attention_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads,
            attention_feedforward_dims=self.attention_feedforward_dims,
            spatial_size=spatial_size,
            z_min_frequency=self.z_min_frequency,
            z_max_frequency=self.z_max_frequency,
        )


"""
Architecture of the image trunk a PosteriorHead reads its latent vectors from.
Spec only holds free hyperparameters, while simulation-derived values (grid size, e.g.)
are passed by the solver at `build` time.
"""
EncoderSpec = Union[
    CnnEncoderSpec,
    SetEncoderSpec,
]

# These dataclasses are frozen and hold numbers, mark them save to serialize so `torch.load` doesn't complain
torch.serialization.add_safe_globals([CnnEncoderSpec, SetEncoderSpec])


def get_default_cnn_encoder_spec(
    spatial_hidden_channels: Sequence[int] = (16, 32),
    embedding_dims: int = 128,
) -> CnnEncoderSpec:
    return CnnEncoderSpec(
        spatial_hidden_channels=spatial_hidden_channels,
        embedding_dims=embedding_dims,
    )


def get_default_set_encoder_spec(
        spatial_hidden_channels: Sequence[int] = (16, 32),
        image_embedding_dims: int = 128,
        z_embedding_dims: int = 128,
        num_attention_layers: int = 4,
        num_attention_heads: int = 8,
        attention_feedforward_dims: int = 512,
        z_min_frequency: float = 0.25,
        z_max_frequency: float = 4.0,
) -> SetEncoderSpec:
    return SetEncoderSpec(
        spatial_hidden_channels=spatial_hidden_channels,
        image_embedding_dims=image_embedding_dims,
        z_embedding_dims=z_embedding_dims,
        num_attention_layers=num_attention_layers,
        num_attention_heads=num_attention_heads,
        attention_feedforward_dims=attention_feedforward_dims,
        z_min_frequency=z_min_frequency,
        z_max_frequency=z_max_frequency,
    )


class CholeskyParameterization(nn.Module):
    """
    Turns a decoder's flattened lower-triangular outputs into a Cholesky factor.
    Used by the full-covariance and mixture-density network posterior heads.
    """

    def __init__(self, dim: int):
        """`dim` = side length of the covariance, i.e. the number of inferred coefficients"""
        super().__init__()
        self.dim = dim
        tril_indices = torch.tril_indices(dim, dim)
        self.register_buffer("tril_indices", tril_indices)
        self.register_buffer("tril_is_diagonal", tril_indices[0] == tril_indices[1])

    @property
    def num_entries(self) -> int:
        """Number of free entries in the lower triangle, i.e. how many outputs to predict"""
        return self.tril_indices.shape[1]

    def initial_bias(self, repeat: int = 1) -> torch.Tensor:
        """
        Outputs an output-head bias that initializes the Cholesky factor to the identity.
        Softplus of the diagonal is 1 and off-diagonals are 0, so training begins from
        unit variance and no correlations.

        Mixture heads use `repeat` to tile across the components.
        """
        inverse_softplus_one = math.log(math.expm1(1.0))  # so softplus of diagonal = 1
        bias = torch.where(self.tril_is_diagonal, inverse_softplus_one, 0.0)
        return bias.repeat(repeat)

    def to_scale_tril(self, chol_entries: torch.Tensor) -> torch.Tensor:
        """
        Converts flattened [..., T] Cholesky factors to lower-triangular [..., dim, dim].
        Also applies softplus to the diagonal, to keep the Cholesky decomposition unique.
        """
        chol_entries = torch.where(self.tril_is_diagonal, F.softplus(chol_entries) + 1e-3, chol_entries)
        scale_tril = chol_entries.new_zeros(*chol_entries.shape[:-1], self.dim, self.dim)
        scale_tril[..., self.tril_indices[0], self.tril_indices[1]] = chol_entries
        return scale_tril


"""
Used by a ZstackSolver to build a PosteriorHead from the coefficient space and a built trunk.
The class itself, or, if a config is needed, the class bound to that config via `functools.partial`.
"""
HeadFactory = Callable[[CoefficientSpace, nn.Module], "PosteriorHead"]


class PosteriorHead(nn.Module, ABC):
    """
    Base class for the posterior predicted by a solver.

    Each subclasses will build a linear projector over an encoder that maps `(images, z)` -> `[B, emb]`,
    interpret its raw output, and define a loss. Subclasses declare their projector widths via
    `out_dims` and (if applicable) `detached_out_dims`.

    If a detached projector head is used, it reads from a detached copy of the latent space, and
    appends its outputs to the main ones along the channel dim. This lets aux outputs (e.g.,
    uncertainty params) train without mucking up the main outputs in the shared trunk.

    Nothing outside the PosteriorHead should try to interpreting the outputs
    of `forward`. Instead, external callers should access `whitened_means`,
    `whitened_distribution`, `whitened_samples`, or `losses`.

    Very specifically does not register the CoefficientSpace as a child, since the
    StackSimulator already owns it, else the state dict would register it twice.
    """

    def __init__(self, coefficients: CoefficientSpace, trunk: nn.Module):
        super().__init__()
        self._coefficients = [coefficients]  # list-wrap so not registered as a child, kind of hacky
        self.trunk = trunk
        self.build_modules()
        self.project = nn.Linear(trunk.embedding_dims, self.out_dims) if self.out_dims else None
        self.detached_project = (
            nn.Linear(trunk.embedding_dims, self.detached_out_dims) if self.detached_out_dims else None
        )
        self.init_projections()

    @property
    def coefficients(self) -> CoefficientSpace:
        """The coefficient space this head predicts over (owned by the simulator)"""
        return self._coefficients[0]

    @property
    @abstractmethod
    def out_dims(self) -> int:
        """Width of the main projection (0 to skip both projections and expose the latent directly)"""
        ...

    @property
    def detached_out_dims(self) -> int:
        """Width of the detached projection (default 0 skips)"""
        return 0

    def build_modules(self) -> None:
        """Hook for subclasses to create modules the projection widths depend on (e.g., a CholeskyParameterization)"""

    def init_projections(self) -> None:
        """Hook for subclasses to seed the projection weights after they're built"""

    def forward(self, images: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Runs the trunk and projection(s) on a [B, Z, H, W] batch of images taken at
        z-positions `z` (ignored by fixed-geometry trunks).
        The results represent different things, depending on the head type.
        """
        latent = self.trunk(images, z)
        if self.project is None:
            return latent
        outputs = self.project(latent)
        if self.detached_project is not None:
            outputs = torch.cat([outputs, self.detached_project(latent.detach())], dim=1)
        return outputs

    @abstractmethod
    def whitened_means(self, encoded: torch.Tensor) -> torch.Tensor:
        """Returns the whitened [B, N_tot] MLE implied by the network output"""
        ...

    @abstractmethod
    def losses(
            self,
            encoded: torch.Tensor,
            targets: torch.Tensor,
            *,
            epoch: int = 0,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns the total loss and a dict of individual losses for whitened targets.
        - epoch: only used by MixtureHead, which may use warmup epochs
        - generator: only used by FlowMatchingHead, which samples inside its loss
        """
        ...

    def val_metrics(self, encoded: torch.Tensor, targets: torch.Tensor) -> dict:
        """Validation-only scalars for logging, subclasses should implement"""
        return {}

    def whitened_distribution(self, encoded: torch.Tensor) -> torch.distributions.Distribution:
        """
        Posterior of non-pinned coefs in whitened space.
        Subclasses that predict a posterior distribution should implement this.

        Losses and metrics will use this during training.
        For inference, callers should use `distribution`.
        """
        raise NotImplementedError(f"{type(self).__name__} has no closed-form posterior density")

    def distribution(self, encoded: torch.Tensor) -> torch.distributions.Distribution:
        """
        Posterior of the non-pinned coefs in physical space.
        Subclasses that predict a posterior distribution should implement this.

        Callers should use this to get the predicted coef distributions.
        Losses and metrics will instead use `whitened_distribution`.
        """
        raise NotImplementedError(f"{type(self).__name__} has no closed-form posterior density")

    def whitened_samples(self, encoded: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Draws [B, S, N_kept] whitened posterior samples.

        Subclasses with a closed-form density can use this for free.
        Others should either override (e.g., flow matching) or inherit the raise above (e.g., MLE).
        """
        return self.whitened_distribution(encoded).sample((num_samples,)).movedim(0, 1)

    def _unwhiten_normal(
            self,
            whitened: torch.distributions.MultivariateNormal,
    ) -> torch.distributions.MultivariateNormal:
        """Maps a whitened multivariate Gaussian over the non-pinned coefs into physical space"""
        scales = self.coefficients.nonpinned_scales
        return torch.distributions.MultivariateNormal(
            whitened.mean * scales + self.coefficients.nonpinned_means,
            scale_tril=scales.unsqueeze(-1) * whitened.scale_tril,
        )

    # Convenience accessors onto the CoefficientSpace, used by subclasses

    @property
    def _keep(self) -> torch.Tensor:
        return ~self.coefficients.is_pinned

    @property
    def _num_kept(self) -> int:
        return self.coefficients.num_nonpinned_coefs


class PointHead(PosteriorHead):
    """
    Pointwise maximum likelihood estimator for each coefficient.
    """

    @property
    def out_dims(self) -> int:
        return self.coefficients.num_coefs

    def whitened_means(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded  # means are the only things predicted

    def losses(
            self,
            encoded: torch.Tensor,
            targets: torch.Tensor,
            *,
            epoch: int = 0,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        keep = self._keep  # mask pinned terms out of the loss
        loss = F.mse_loss(encoded[:, keep], targets[:, keep])
        return loss, {}


class HeteroscedasticHead(PosteriorHead):
    """
    Predicts the most likely aberration coefficients along with their individual uncertainties.
    Each coefficient is modeled as its own Gaussian, so correlations and multimodality are still not handled.
    """

    @property
    def out_dims(self) -> int:
        return 2 * self.coefficients.num_coefs  # mu and log(sigma) each

    def _whitened_normal(self, encoded: torch.Tensor) -> torch.distributions.Normal:
        """Reads the raw decoder output as a per-coefficient Gaussian in whitened space"""
        mu, log_sigma = encoded.chunk(2, dim=1)
        return torch.distributions.Normal(mu, log_sigma.exp())

    def whitened_means(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded.chunk(2, dim=1)[0]  # means, no stdevs

    def losses(
            self,
            encoded: torch.Tensor,
            targets: torch.Tensor,
            *,
            epoch: int = 0,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        dist = self._whitened_normal(encoded)
        keep = self._keep  # mask pinned terms out of the losses
        mu_loss = F.mse_loss(dist.mean[:, keep], targets[:, keep])
        sigma_loss = F.gaussian_nll_loss(
            dist.mean.detach()[:, keep], targets[:, keep], dist.variance[:, keep]
        )  # detach the mean so the sigma head can't push it back to the uninformative priors
        return mu_loss + sigma_loss, {"mu_mse": mu_loss, "sigma_nll": sigma_loss}

    def val_metrics(self, encoded: torch.Tensor, targets: torch.Tensor) -> dict:
        dist = self._whitened_normal(encoded)
        z_scores = ((dist.mean - targets) / dist.stddev)[:, self._keep]
        metrics = {"z_std": z_scores.std()}  # ~1 when the sigmas are calibrated
        return metrics

    def whitened_distribution(self, encoded: torch.Tensor) -> torch.distributions.Independent:
        """
        Joint distribution over non-pinned coefs is diagonal, but wrap it as the marginal
        of a full pairwise multivariate distribution for API consistency with other heads.
        """
        dist = self._whitened_normal(encoded)
        keep = self._keep
        marginals = torch.distributions.Normal(dist.mean[:, keep], dist.stddev[:, keep])
        return torch.distributions.Independent(marginals, reinterpreted_batch_ndims=1)

    def distribution(self, encoded: torch.Tensor) -> torch.distributions.Independent:
        """Unwhitens the diagonal Gaussian into physical coefficient space"""
        whitened = self.whitened_distribution(encoded).base_dist
        scales, means = self.coefficients.nonpinned_scales, self.coefficients.nonpinned_means
        marginals = torch.distributions.Normal(
            whitened.mean * scales + means, whitened.stddev * scales
        )
        return torch.distributions.Independent(marginals, reinterpreted_batch_ndims=1)


class CovarianceHead(PosteriorHead):
    """
    Predicts the coefficients along with their full covariance matrix.

    Handles correlations (at least, pairwise ones), but still not multimodality.

    The covariance is represented by its Cholesky factor, flattened and predicted by
    an extra, detached, projector.
    """

    def build_modules(self) -> None:
        self.cholesky = CholeskyParameterization(self.coefficients.num_nonpinned_coefs)

    @property
    def out_dims(self) -> int:
        return self.coefficients.num_coefs

    @property
    def detached_out_dims(self) -> int:
        return self.cholesky.num_entries

    def init_projections(self) -> None:
        """Initializes the detached projection to 0 covariance"""
        nn.init.zeros_(self.detached_project.weight)
        with torch.no_grad():
            self.detached_project.bias.copy_(self.cholesky.initial_bias())

    def whitened_means(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded[:, :self.coefficients.num_coefs]  # means, no covariance

    def whitened_distribution(self, encoded: torch.Tensor) -> torch.distributions.MultivariateNormal:
        """Reads the raw decoder output as a joint Gaussian in whitened space"""
        means = self.whitened_means(encoded)[:, self._keep]
        scale_tril = self.cholesky.to_scale_tril(encoded[:, self.coefficients.num_coefs:])
        return torch.distributions.MultivariateNormal(means, scale_tril=scale_tril)

    def distribution(self, encoded: torch.Tensor) -> torch.distributions.MultivariateNormal:
        """Unwhitens the joint Gaussian into physical coefficient space"""
        return self._unwhiten_normal(self.whitened_distribution(encoded))

    def losses(
            self,
            encoded: torch.Tensor,
            targets: torch.Tensor,
            *,
            epoch: int = 0,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        dist = self.whitened_distribution(encoded)
        targets_kept = targets[:, self._keep]
        mu_loss = F.mse_loss(dist.mean, targets_kept)
        detached_dist = torch.distributions.MultivariateNormal(
            dist.mean.detach(), scale_tril=dist.scale_tril
        )  # detach the mean so the covariance head can't push it back to the uninformative priors
        sigma_loss = -detached_dist.log_prob(targets_kept).mean() / self._num_kept  # per-coefficient scale
        return mu_loss + sigma_loss, {"mu_mse": mu_loss, "sigma_nll": sigma_loss}

    def val_metrics(self, encoded: torch.Tensor, targets: torch.Tensor) -> dict:
        dist = self.whitened_distribution(encoded)
        residuals = (targets[:, self._keep] - dist.mean).unsqueeze(-1)
        z_scores = torch.linalg.solve_triangular(dist.scale_tril, residuals, upper=False)
        metrics = {"z_std": z_scores.std()}  # ~1 when the covariances are calibrated
        return metrics


class MixtureHead(PosteriorHead):
    """
    Predicts a Gaussian mixture over the coefficients. Finally handles multimodality!

    Models up to `num_components` components, each parameterized like the single Gaussian
    of `CovarianceHead` (i.e., component mean + flattened Cholesky factor), joined
    by a mixing coefficient for each component.

    `mixing_warmup_epochs` holds the mixing weights uniform and frozen for the
    first epochs so the component means can specialize before the weights are learned.
    Without it, multimodal targets tend to collapse onto a single component early.

    `mixing_entropy_weight` weights the entropy loss between the mixing logits
    to further discourage comopnent collapse.
    """

    def __init__(
            self,
            coefficients: CoefficientSpace,
            trunk: nn.Module,
            *,
            cfg: MixtureConfig,
    ):
        self.num_components = cfg.num_components
        self.mixing_warmup_epochs = cfg.mixing_warmup_epochs
        self.min_allocation = cfg.min_allocation
        self.mixing_entropy_weight = cfg.mixing_entropy_weight
        super().__init__(coefficients, trunk)

    def build_modules(self) -> None:
        self.cholesky = CholeskyParameterization(self.coefficients.num_nonpinned_coefs)

    @property
    def out_dims(self) -> int:
        return self.num_components * self.coefficients.num_coefs  # per-component means

    @property
    def detached_out_dims(self) -> int:
        return self.num_components + self.num_components * self.cholesky.num_entries  # logits + Cholesky factors

    def init_projections(self) -> None:
        """
        Initializes the detached projection to uniform mixing and 0 covariance.
        Seeds the means with an O(1) spread to encourage specialization.
        """
        nn.init.zeros_(self.detached_project.weight)
        with torch.no_grad():
            logit_bias = torch.zeros(self.num_components)
            self.detached_project.bias.copy_(
                torch.cat([logit_bias, self.cholesky.initial_bias(self.num_components)])
            )

            generator = torch.Generator().manual_seed(0)
            mean_bias = torch.randn(self.num_components, self.coefficients.num_coefs, generator=generator)
            self.project.bias.copy_(mean_bias.flatten())  # seed to encourage specialization

    def split_predictions(
            self,
            encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method, splits the raw [B, D] decoder output into:
        - means: [B, K, N] (in whitened space)
        - logits: [B, K] mixing logits
        - chol_entries: [B, K, T] flattened Cholesky factors
        """
        num_means = self.num_components * self.coefficients.num_coefs
        means = encoded[:, :num_means].reshape(-1, self.num_components, self.coefficients.num_coefs)
        logits = encoded[:, num_means:num_means + self.num_components]
        chol_entries = encoded[:, num_means + self.num_components:].reshape(
            -1, self.num_components, self.cholesky.num_entries
        )
        return means, logits, chol_entries

    def whitened_means(self, encoded: torch.Tensor) -> torch.Tensor:
        """Whitened means of the highest-weight component"""
        means, logits, _ = self.split_predictions(encoded)
        batch_idx = torch.arange(means.shape[0], device=means.device)
        return means[batch_idx, logits.argmax(dim=1)]

    def whitened_distribution(self, encoded: torch.Tensor) -> torch.distributions.MixtureSameFamily:
        """Reads the raw decoder output as a Gaussian mixture in whitened space (non-pinned coefs only)"""
        means, logits, chol_entries = self.split_predictions(encoded)
        components = torch.distributions.MultivariateNormal(
            means[..., self._keep], scale_tril=self.cholesky.to_scale_tril(chol_entries)
        )
        weights = torch.distributions.Categorical(logits=logits)
        return torch.distributions.MixtureSameFamily(weights, components)

    def distribution(self, encoded: torch.Tensor) -> torch.distributions.MixtureSameFamily:
        """Unwhitens the mixture into physical coefficient space, weights unchanged"""
        whitened = self.whitened_distribution(encoded)
        return torch.distributions.MixtureSameFamily(
            whitened.mixture_distribution,
            self._unwhiten_normal(whitened.component_distribution),
        )

    def _mixing_logits(self, logits: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Mixing logits used by the losses.

        During the mixing warmup epochs, the learned logits are replaced with uniform constants
        to keep them frozen. This lets the components specialize before the mixture weights
        are learned, so multimodal targets don't collapse too early.
        """
        logits = torch.distributions.Categorical(logits=logits).logits
        if epoch < self.mixing_warmup_epochs:
            return torch.zeros_like(logits)
        return logits

    def losses(
            self,
            encoded: torch.Tensor,
            targets: torch.Tensor,
            *,
            epoch: int = 0,
            generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        means, raw_logits, chol_entries = self.split_predictions(encoded)
        scale_tril = self.cholesky.to_scale_tril(chol_entries)
        dist = torch.distributions.MultivariateNormal(means[..., self._keep], scale_tril=scale_tril)
        mixing_logits = self._mixing_logits(raw_logits, epoch)
        targets_kept = targets[:, self._keep]

        with torch.no_grad():
            allocations = dist.log_prob(targets_kept.unsqueeze(1)).softmax(dim=1)
            floor = self.min_allocation
            if floor > 0 and epoch >= self.mixing_warmup_epochs:
                allocations = (1 - floor * self.num_components) * allocations + floor

        dist_mse = (dist.mean - targets_kept.unsqueeze(1)).square().mean(dim=2)
        mu_loss = (allocations * dist_mse).sum(dim=1).mean()  # allocation-weighted MSE

        detached_mixture = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=mixing_logits),
            torch.distributions.MultivariateNormal(
                dist.mean.detach(), scale_tril=dist.scale_tril
            ),
        )  # detach the mean so the covariance head can't push it back to the uninformative priors
        sigma_nll = -detached_mixture.log_prob(targets_kept).mean() / self._num_kept  # per-coefficient scale

        weight_entropy = torch.distributions.Categorical(
            logits=mixing_logits
        ).entropy().mean()  # 0 if single component, log(K) else
        logs = {"mu_mse": mu_loss, "sigma_nll": sigma_nll, "weight_entropy": weight_entropy}
        loss = mu_loss + sigma_nll - self.mixing_entropy_weight * weight_entropy
        return loss, logs

    def val_metrics(self, encoded: torch.Tensor, targets: torch.Tensor) -> dict:
        metrics = {}
        means, logits, _ = self.split_predictions(encoded)

        max_allocations = logits.softmax(dim=1).amax(dim=1)
        metrics["dominant_component_allocation"] = max_allocations.mean()

        residuals = means[..., self._keep] - targets[:, self._keep].unsqueeze(1)
        component_mse = residuals.square().mean(dim=2)  # [B, K]
        batch_idx = torch.arange(means.shape[0], device=means.device)
        metrics["dominant_component_rmse"] = component_mse[batch_idx, logits.argmax(dim=1)].mean().sqrt()
        metrics["best_component_rmse"] = component_mse.amin(dim=1).mean().sqrt()

        return metrics
