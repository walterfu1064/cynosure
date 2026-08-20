from typing import Optional

import numpy as np
import torch

from cynosure.zstack_decoding.noise_model import scale_to_photon_counts
from cynosure.zstack_decoding.stack_simulator import StackSimulator


def _resolve_noise_parameters(
        simulator: StackSimulator,
        photons: Optional[float],
        background: Optional[float],
) -> tuple[float, float, float]:
    """Resolves (photons, background, read noise), falling back to the `NoiseConfig` averages"""
    if simulator.noise_cfg:
        cfg = simulator.noise_cfg
        defaults = (cfg.average_photons, cfg.average_background, cfg.read_noise)
    else:
        defaults = (1, 0, 0)
    return photons or defaults[0], background or defaults[1], defaults[2]


def _fisher_from_jacobian(
        counts: torch.Tensor,
        jacobian: torch.Tensor,
        read_noise: float,
) -> torch.Tensor:
    """
    Calculates a Fisher matrix from a z-stack Jacobian, under approximately Gaussian shot + read noise.

    Arguments:
    - counts: [Z, H, W] expected photon counts
    - jacobian: [Z, H, W, C] derivative of `counts` with respect to each parameter
    Returns:
    - [Z, C, C]
    """
    inv_variance = 1.0 / (counts + read_noise ** 2)  # [Z, H, W]
    return torch.einsum("zhw,zhwi,zhwj->zij", inv_variance, jacobian, jacobian)


def simulate_image_as_photon_counts(
        coefs: torch.Tensor,
        z: torch.Tensor,
        photons: float,
        background: float,
        simulator: StackSimulator,
) -> torch.Tensor:
    """
    Forwards-simulates a z-stack and normalizes it to raw photon counts.

    Arguments:
    - coefs: [C_kept,], the non-pinned coefficients, ordered as from `CoefficientSpace.gather_nonpinned()`
    - z: [Z,]
    - photons, background: scalars
    Returns:
    - [Z, H, W]
    """
    num_z = z.shape[0]

    coefficients = simulator.coefficients
    full_coefs = coefficients.target_means + coefficients.nonpinned_basis @ coefs

    defocus = simulator.propagator.defocus_from_objective_z(z)
    phase_coefs = coefficients.block_coefs(full_coefs, "phase")
    amp_coefs = coefficients.block_coefs(full_coefs, "amp")
    phase_coefs = phase_coefs.unsqueeze(0).expand(num_z, -1)
    amp_coefs = amp_coefs.unsqueeze(0).expand(num_z, -1)
    psf = simulator.propagator(defocus, phase_coefs, amp_coefs)  # [Z, H, W]
    images = simulator.object_distribution(psf.unsqueeze(0), batch_size=1).squeeze(0)
    return scale_to_photon_counts(images, photons, background)


def calculate_image_jacobian(
        coefs: torch.Tensor,
        z: torch.Tensor,
        num_photons: float,
        background_level: float,
        simulator: StackSimulator,
) -> torch.Tensor:
    """
    Jacobian of the simulated z-stack with respect to each non-pinned aberration coefficient.

    Arguments:
    - coefs: [C_kept,], the non-pinned coefficients, ordered as from `CoefficientSpace.gather_nonpinned()`
    - z: [Z,]
    - num_photons, background_level: scalars
    Returns:
    - [Z, H, W, C_kept]
    """
    jac = torch.func.jacfwd(simulate_image_as_photon_counts)(coefs, z, num_photons, background_level, simulator)
    return jac


def calculate_defocus_derivative(
        coefs: torch.Tensor,
        z: torch.Tensor,
        num_photons: float,
        background_level: float,
        simulator: StackSimulator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates a z-stack, then calculates its derivative with respect to each plane's defocus.
    JVP with a vector of ones is more efficient than a full `jacfwd` for this term.

    Arguments:
    - coefs: [C_kept,], the non-pinned coefficients, ordered as from `CoefficientSpace.gather_nonpinned()`
    - z: [Z,]
    - photons, background: scalars
    Returns:
    - simulated z-stack, [Z, H, W]
    - its derivative with respect to objective z, [Z, H, W]
    """
    def simulate(plane_z: torch.Tensor) -> torch.Tensor:
        return simulate_image_as_photon_counts(coefs, plane_z, num_photons, background_level, simulator)
    return torch.func.jvp(simulate, (z,), (torch.ones_like(z),))


def calculate_fisher_matrix(
        coefs: torch.Tensor,
        z: torch.Tensor,
        simulator: StackSimulator,
        photons: Optional[float] = None,
        background: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates a z-stack, then calculates its Fisher matrix with respect to the non-pinned aberration coefficients.

    Arguments:
    - coefs: [C_kept,], the free (non-pinned) coefficients, as from `CoefficientSpace.gather_nonpinned()`
    - z: [Z,]
    - photons, background: scalars, use NoiseCfg averages if None
    Returns:
    - simulated z-stack, [Z, H, W]
    - Fisher matrix, [Z, C_kept, C_kept]
    """
    photons, background, read_noise = _resolve_noise_parameters(simulator, photons, background)
    counts = simulate_image_as_photon_counts(coefs, z, photons, background, simulator)
    jac = calculate_image_jacobian(coefs, z, photons, background, simulator)
    return counts, _fisher_from_jacobian(counts, jac, read_noise)  # [Z, C_kept, C_kept]


def calculate_augmented_fisher_matrix(
        coefs: torch.Tensor,
        z: torch.Tensor,
        simulator: StackSimulator,
        photons: Optional[float] = None,
        background: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates a z-stack, then calculates its Fisher matrix with respect to both the non-pinned aberration
    coefficients and the defocus of each plane (in the final row+column).

    Defocus can be equivalently expressed in terms of the m=0 phase coefficients
    (see `ZstackSolver._z_jitter_to_phase_coefs()`). Pulling it out explicitly lets us
    disantangle it from the "actual" m=0 aberrations.

    Arguments:
    - coefs: [C_kept,], the free (non-pinned) coefficients, as from `CoefficientSpace.gather_nonpinned()`
    - z: [Z,]
    - photons, background: scalars, use NoiseCfg averages if None
    Returns:
    - simulated z-stack, [Z, H, W]
    - Fisher matrix, [Z, C_kept+1, C_kept+1] (defocus as the last row+column)
    """
    photons, background, read_noise = _resolve_noise_parameters(simulator, photons, background)
    coef_jacobian = calculate_image_jacobian(coefs, z, photons, background, simulator)  # [Z, H, W, C_kept]
    counts, defocus_derivative = calculate_defocus_derivative(coefs, z, photons, background, simulator)
    jacobian = torch.cat([coef_jacobian, defocus_derivative.unsqueeze(-1)], dim=-1)  # [Z, H, W, C_kept+1]
    return counts, _fisher_from_jacobian(counts, jacobian, read_noise)


def defocus_direction(simulator: StackSimulator) -> torch.Tensor:
    """
    Returns the [C_kept,] non-pinned phase aberrations equivalent to 1 um of objective-z displacement.
    I.e., the "z" unit vector in phase aberration space.
    Matches `StackSimulator.z_offset_to_coefficients`, restricted to the non-pinned phase coefficients.
    """
    coefficients = simulator.coefficients
    dtype = simulator.ftype
    device = simulator.device
    one = torch.tensor(1, dtype=dtype, device=device)  # less than 2, but a whole lot more than 0
    phase_direction = simulator.propagator.defocus_from_objective_z(one) * simulator.defocus_phase_coefs
    blocks = [  # every other block is zero, but still has to be padded out for the join-gather
        phase_direction if block.name == "phase"
        else torch.zeros(block.size, dtype=dtype, device=device)
        for block in coefficients.blocks
    ]
    return coefficients.gather_nonpinned(coefficients.join(*blocks))


def coefficient_prior_covariance(simulator: StackSimulator, include_jitter: bool = True) -> np.ndarray:
    """
    Returns the [C_kept, C_kept] prior covariance over the non-pinned coefficients.

    The prior is diagonal (see `PriorConfig`), but z-jitter couples across the m=0 coefficients.
    If `include_jitter=True`, adds those terms to the covariance, along the diagonal.
    Else, models the aberrations alone, with the z-positions taken to be known.
    """
    coefficients = simulator.coefficients
    sigmas = coefficients.gather_nonpinned(coefficients.prior_scales).numpy(force=True)  # not jitter-widened yet
    covariance = np.diag(sigmas ** 2)

    if include_jitter and simulator.z_jitter.variance > 0:
        direction = defocus_direction(simulator).numpy(force=True)
        covariance = covariance + simulator.z_jitter.variance * np.outer(direction, direction)

    return covariance


def augmented_prior_covariance(simulator: StackSimulator) -> np.ndarray:
    """
    Extends the output of `coefficient_prior_covariance`, adding z along the last row+column.
    Returns as a [C_kept+1, C_kept+1] covariance matrix matching `calculate_augmented_fisher_matrix()`.
    """
    if simulator.z_jitter.variance <= 0:
        raise ValueError("no z-jitter, so z is exactly known, use `coefficient_prior_covariance`")

    num_coefs = simulator.coefficients.num_nonpinned_coefs
    covariance = np.zeros((num_coefs + 1, num_coefs + 1))
    covariance[:num_coefs, :num_coefs] = coefficient_prior_covariance(simulator, include_jitter=False)
    covariance[num_coefs, num_coefs] = simulator.z_jitter.variance
    return covariance


def calculate_mutual_information(
        fisher_matrix: np.ndarray,
        prior_covariance: np.ndarray,
        indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Gaussian mutual information (in bits) between the parameters and an observed z-stack.
    If `indices` is passed, marginalizes over all parameters not included in `indices`.
    Should be batch-averaged in the output bits, not in the input Fisher matrices.

    Arguments:
    - fisher_matrix_samples: [..., C, C]
    - prior_covariance: [C, C]
    - indices: [S,] indices of the parameters to keep
    Returns:
    - information in bits, [...,]
    """
    if indices is None:
        indices = np.arange(fisher_matrix.shape[-1])
    rows, cols = indices[:, None], indices[None, :]
    prior_precision = np.linalg.inv(prior_covariance)
    posterior_covariance = np.linalg.inv(fisher_matrix + prior_precision)
    _, prior_log_det = np.linalg.slogdet(prior_covariance[rows, cols])
    _, posterior_log_det = np.linalg.slogdet(posterior_covariance[..., rows, cols])
    return 0.5 * (prior_log_det - posterior_log_det) / np.log(2)


def calculate_van_trees_bounds(
        fisher_matrix: np.ndarray,
        prior_covariance: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns the van Trees bounds for the parameters in a Fisher matrix, optionally
    under a Gaussian prior. If no prior is given, defaults to the uninformative prior.
    Batch-averaging should be done to the Fisher matrices beforehand, not to the
    van Trees bounds afterwards.

    Arguments:
    - fisher_matrix: [..., C, C]
    - prior_covariance: [C, C] covariance matrix
    Returns:
    - standard deviation bound per parameter, [..., C]
    """
    if prior_covariance is None:
        prior_precision = np.zeros_like(fisher_matrix)
    else:
        prior_precision = np.linalg.inv(prior_covariance)
    posterior_covariance = np.linalg.inv(fisher_matrix + prior_precision)
    return np.sqrt(np.diagonal(posterior_covariance, axis1=-2, axis2=-1))
