"""
Output heads that predict the coefficient posteriors in various ways.

Each PosteriorHead builds its own network, decides how to interpret that network's
outputs, and handles its own loss calculations and validation metrics.

Bit of hackiness: each PosteriorHead holds a reference to the CoefficientSpace
it predicts over, but deliberately doesn't register it as a child module, since
the StackSimulator owns it and will already include it in the state dict.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coefficients import CoefficientSpace
from .submodules import CnnDecoder, CnnDecoder_DetachedHead


@dataclass(frozen=True)
class EncoderSpec:
    """Architecture of the image encoder that every PosteriorHead builds from"""
    in_channels: int  # z-planes per stack
    spatial_hidden_channels: Sequence[int]
    embedding_dims: int
    spatial_size: int  # object grid side length


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


class PosteriorHead(nn.Module, ABC):
    """
    Base class for the posterior predicted by a solver.

    Each subclasses will build a network, interpret its raw output, and define a loss.

    Nothing outside the PosteriorHead should try to interpreting the outputs
    of `forward`. Instead, external callers should access `whitened_means`,
    `whitened_distribution`, or `losses`.

    Very specifically does not register the CoefficientSpace as a child, since the
    StackSimulator already owns it, else the state dict would register it twice.
    """

    def __init__(self, coefficients: CoefficientSpace, encoder: EncoderSpec):
        super().__init__()
        self._coefficients = [coefficients]  # list-wrap so not registered as a child, kind of hacky
        self.encoder = encoder
        self.decoder = self.build_network(encoder)

    @property
    def coefficients(self) -> CoefficientSpace:
        """The coefficient space this head predicts over (owned by the simulator)"""
        return self._coefficients[0]

    @abstractmethod
    def build_network(self, encoder: EncoderSpec) -> nn.Module:
        """Builds the network this head reads from"""
        ...

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Runs the network on a [B, num_z, H, W] batch of images.
        The results represent different things, depending on the head type.
        """
        return self.decoder(images)

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
        """Posterior of non-pinned coefs in whitened space; subclasses should implement if exists"""
        raise NotImplementedError(f"{type(self).__name__} has no closed-form posterior density")

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

    def build_network(self, encoder: EncoderSpec) -> nn.Module:
        return CnnDecoder(
            in_channels=encoder.in_channels,
            spatial_hidden_channels=encoder.spatial_hidden_channels,
            embedding_dims=encoder.embedding_dims,
            out_dims=self.coefficients.num_coefs,
            spatial_size=encoder.spatial_size,
        )

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

    def build_network(self, encoder: EncoderSpec) -> nn.Module:
        return CnnDecoder(
            in_channels=encoder.in_channels,
            spatial_hidden_channels=encoder.spatial_hidden_channels,
            embedding_dims=encoder.embedding_dims,
            out_dims=2 * self.coefficients.num_coefs,  # mu and log(sigma) each
            spatial_size=encoder.spatial_size,
        )

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


class CovarianceHead(PosteriorHead):
    """
    Predicts the coefficients along with their full covariance matrix.

    Handles correlations (at least, pairwise ones), but still not multimodality.

    The covariance is represented by its Cholesky factor, flattened and predicted by
    an extra, detached, decoder head.
    """

    def build_network(self, encoder: EncoderSpec) -> nn.Module:
        """
        Builds a decoder with a detached head for the Cholesky factor.
        Initializes the detached head to 0 covariance.
        """
        self.cholesky = CholeskyParameterization(self.coefficients.num_nonpinned_coefs)

        decoder = CnnDecoder_DetachedHead(
            in_channels=encoder.in_channels,
            spatial_hidden_channels=encoder.spatial_hidden_channels,
            embedding_dims=encoder.embedding_dims,
            out_dims=self.coefficients.num_coefs,
            detached_out_dims=self.cholesky.num_entries,
            spatial_size=encoder.spatial_size,
        )

        nn.init.zeros_(decoder.detached_head.weight)
        with torch.no_grad():
            decoder.detached_head.bias.copy_(self.cholesky.initial_bias())
        return decoder

    def whitened_means(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded[:, :self.coefficients.num_coefs]  # means, no covariance

    def whitened_distribution(self, encoded: torch.Tensor) -> torch.distributions.MultivariateNormal:
        """Reads the raw decoder output as a joint Gaussian in whitened space"""
        means = self.whitened_means(encoded)[:, self._keep]
        scale_tril = self.cholesky.to_scale_tril(encoded[:, self.coefficients.num_coefs:])
        return torch.distributions.MultivariateNormal(means, scale_tril=scale_tril)

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
            encoder: EncoderSpec,
            *,
            num_components: int = 4,
            mixing_warmup_epochs: int = 0,
            min_allocation: float = 0.0,
            mixing_entropy_weight: float = 0.0,
    ):
        self.num_components = num_components
        self.mixing_warmup_epochs = mixing_warmup_epochs
        self.min_allocation = min_allocation
        self.mixing_entropy_weight = mixing_entropy_weight
        super().__init__(coefficients, encoder)

    def build_network(self, encoder: EncoderSpec) -> nn.Module:
        """
        Builds a decoder with a detached head for the Cholesky factor and mixing logits.
        Initializes the detached head to 0 covariance.
        Seeds the means with an O(1) spread to encourage specialization.
        """
        self.cholesky = CholeskyParameterization(self.coefficients.num_nonpinned_coefs)

        decoder = CnnDecoder_DetachedHead(
            in_channels=encoder.in_channels,
            spatial_hidden_channels=encoder.spatial_hidden_channels,
            embedding_dims=encoder.embedding_dims,
            out_dims=self.num_components * self.coefficients.num_coefs,
            detached_out_dims=self.num_components + self.num_components * self.cholesky.num_entries,
            spatial_size=encoder.spatial_size,
        )

        nn.init.zeros_(decoder.detached_head.weight)
        with torch.no_grad():
            logit_bias = torch.zeros(self.num_components)
            decoder.detached_head.bias.copy_(
                torch.cat([logit_bias, self.cholesky.initial_bias(self.num_components)])
            )

            generator = torch.Generator().manual_seed(0)
            mean_bias = torch.randn(self.num_components, self.coefficients.num_coefs, generator=generator)
            decoder.project.bias.copy_(mean_bias.flatten())  # seed to encourage specialization
        return decoder

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


class FlowMatchingHead(PosteriorHead):
    """
    Predicts a conditional velocity field that transports a standard normal distribution
    into the posterior over aberration coefficients, given a z-stack.

    Unlike the mixture model, the posterior is represented implicitly: there is no closed-form
    density, only the ability to draw samples by integrating the learned ODE. In exchange the
    training objective is a plain regression, so none of the mixture's failure modes
    (collapsing weights, covariance-preconditioned mean gradients) apply.

    The flow lives in whitened space over the non-piston coefficients, where the
    sampling prior is already close to a standard normal, to make its job easier.
    """
    def __init__(self):
        raise NotImplementedError("I don't want to deal with this right now")

