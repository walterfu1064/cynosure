from typing import Optional

import numpy as np
import torch

from cynosure.zstack_decoding import ZstackSolver
from cynosure.utilities.fft_utilities import convolve_psf_with_object
from cynosure.zstack_decoding.noise_model import scale_to_photon_counts


def nonpiston_basis(solver: ZstackSolver) -> torch.Tensor:
    """
    Returns a [C, C_kept] matrix scattering the non-piston coefs back into a full coef vector.

    Full coefs can be obtained using `nonpiston_basis(solver) @ nonpiston_coefs`.
    Needs to be matmul instead of masked write so it can batch in `vmap` which `jacfwd` uses.

    TODO - add to ZstackSolver proper if this becomes more widely useful
    """
    identity = torch.eye(solver.num_coefs, dtype=solver.ftype, device=solver.device)
    return identity[:, ~solver.is_piston]


def _resolve_noise_parameters(
        solver: ZstackSolver,
        photons: Optional[float],
        background: Optional[float],
) -> tuple[float, float, float]:
    """Resolves (photons, background, read noise), falling back to the solver's `NoiseConfig` averages"""
    if solver.noise_cfg:
        defaults = (solver.noise_cfg.average_photons, solver.noise_cfg.average_background, solver.noise_cfg.read_noise)
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
        solver: ZstackSolver,
) -> torch.Tensor:
    """
    Forwards-simulates a z-stack and normalizes it to raw photon counts.

    Arguments:
    - coefs: [C_kept,], the non-piston coefficients, ordered as from `ZstackSolver.gather_nonpiston()`
    - z: [Z,]
    - photons, background: scalars
    Returns:
    - [Z, H, W]
    """
    num_z = z.shape[0]

    full_coefs = solver.target_means + nonpiston_basis(solver) @ coefs  # target_means, not zeros as `scatter_nonpiston`

    defocus = solver.propagator.defocus_from_objective_z(z)
    phase_coefs, amp_coefs = solver.split_phase_amp(full_coefs)
    phase_coefs = phase_coefs.unsqueeze(0).expand(num_z, -1)
    amp_coefs = amp_coefs.unsqueeze(0).expand(num_z, -1)
    psf = solver.propagator(defocus, phase_coefs, amp_coefs)
    images = convolve_psf_with_object(solver.object_distribution, psf)
    return scale_to_photon_counts(images, photons, background)


def calculate_image_jacobian(
        coefs: torch.Tensor,
        z: torch.Tensor,
        num_photons: float,
        background_level: float,
        solver: ZstackSolver,
) -> torch.Tensor:
    """
    Jacobian of the simulated z-stack with respect to each non-piston aberration coefficient.

    Arguments:
    - coefs: [C_kept,], the non-piston coefficients, ordered as from `ZstackSolver.gather_nonpiston()`
    - z: [Z,]
    - num_photons, background_level: scalars
    Returns:
    - [Z, H, W, C_kept]
    """
    jac = torch.func.jacfwd(simulate_image_as_photon_counts)(coefs, z, num_photons, background_level, solver)
    return jac


def calculate_defocus_derivative(
        coefs: torch.Tensor,
        z: torch.Tensor,
        num_photons: float,
        background_level: float,
        solver: ZstackSolver,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates a z-stack, then calculates its derivative with respect to each plane's defocus.
    JVP with a vector of ones is more efficient than a full `jacfwd` for this term.

    Arguments:
    - coefs: [C_kept,], the non-piston coefficients, ordered as from `ZstackSolver.gather_nonpiston()`
    - z: [Z,]
    - photons, background: scalars
    Returns:
    - simulated z-stack, [Z, H, W]
    - its derivative with respect to objective z, [Z, H, W]
    """
    def simulate(plane_z: torch.Tensor) -> torch.Tensor:
        return simulate_image_as_photon_counts(coefs, plane_z, num_photons, background_level, solver)
    return torch.func.jvp(simulate, (z,), (torch.ones_like(z),))


def calculate_fisher_matrix(
        coefs: torch.Tensor,
        z: torch.Tensor,
        solver: ZstackSolver,
        photons: Optional[float] = None,
        background: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates a z-stack, then calculates its Fisher matrix with respect to the non-piston aberration coefficients.

    Arguments:
    - coefs: [C_kept,], the free (non-piston) coefficients, as from `ZstackSolver.gather_nonpiston()`
    - z: [Z,]
    - photons, background: scalars, use NoiseCfg averages if None
    Returns:
    - simulated z-stack, [Z, H, W]
    - Fisher matrix, [Z, C_kept, C_kept]
    """
    photons, background, read_noise = _resolve_noise_parameters(solver, photons, background)
    counts = simulate_image_as_photon_counts(coefs, z, photons, background, solver)
    jac = calculate_image_jacobian(coefs, z, photons, background, solver)
    return counts, _fisher_from_jacobian(counts, jac, read_noise)  # [Z, C_kept, C_kept]


def calculate_augmented_fisher_matrix(
        coefs: torch.Tensor,
        z: torch.Tensor,
        solver: ZstackSolver,
        photons: Optional[float] = None,
        background: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates a z-stack, then calculates its Fisher matrix with respect to both the non-piston aberration
    coefficients and the defocus of each plane (in the final row+column).

    Defocus can be equivalently expressed in terms of the m=0 phase coefficients
    (see `ZstackSolver._z_jitter_to_phase_coefs()`). Pulling it out explicitly lets us
    disantangle it from the "actual" m=0 aberrations.

    Arguments:
    - coefs: [C_kept,], the free (non-piston) coefficients, as from `ZstackSolver.gather_nonpiston()`
    - z: [Z,]
    - photons, background: scalars, use NoiseCfg averages if None
    Returns:
    - simulated z-stack, [Z, H, W]
    - Fisher matrix, [Z, C_kept+1, C_kept+1] (defocus as the last row+column)
    """
    photons, background, read_noise = _resolve_noise_parameters(solver, photons, background)
    coef_jacobian = calculate_image_jacobian(coefs, z, photons, background, solver)  # [Z, H, W, C_kept]
    counts, defocus_derivative = calculate_defocus_derivative(coefs, z, photons, background, solver)
    jacobian = torch.cat([coef_jacobian, defocus_derivative.unsqueeze(-1)], dim=-1)  # [Z, H, W, C_kept+1]
    return counts, _fisher_from_jacobian(counts, jacobian, read_noise)


def defocus_direction(solver: ZstackSolver) -> torch.Tensor:
    """
    Returns the [C_kept,] non-piston phase aberrations equivalent to 1 um of objective-z displacement.
    I.e., the "z" unit vector in phase aberration space.
    Matches `ZstackSolver._z_jitter_to_phase_coefs`, restricted to the non-piston phase coefficients.
    """
    dtype = solver.ftype
    device = solver.device
    one = torch.tensor(1, dtype=dtype, device=device)  # less than 2, but a whole lot more than 0
    phase_direction = solver.propagator.defocus_from_objective_z(one) * solver.defocus_phase_coefs
    amp_direction = torch.zeros(solver.num_amp_coefs, dtype=dtype, device=device)  # just to pad the join-gather
    return solver.gather_nonpiston(solver.join_phase_amp(phase_direction, amp_direction))


def _uniform_z_jitter_variance(solver: ZstackSolver) -> float:
    """Variance of the z-jitter, which is uniform over [-z_jitter, +z_jitter]"""
    return solver.z_jitter ** 2 / 3


def coefficient_prior_covariance(solver: ZstackSolver, include_jitter: bool = True) -> np.ndarray:
    """
    Returns the [C_kept, C_kept] prior covariance over the non-piston coefficients.

    The prior is diagonal (see `PriorConfig`), but z-jitter couples across the m=0 coefficients.
    If `include_jitter=True`, adds those terms to the covariance, along the diagonal.
    Else, models the aberrations alone, with the z-positions taken to be known.

    TODO - This duplicates parts of `ZstackSolver._setup_whitening()`. Probably refactor and consolidate.
    """

    phase_nm = solver.propagator.phase_projector.nm_indices
    phase_scales = solver.phase_prior_cfg.coef_scales(phase_nm)
    amp_nm = solver.propagator.amp_projector.nm_indices
    amp_scales = solver.amp_prior_cfg.coef_scales(amp_nm)
    scales = torch.cat([phase_scales, amp_scales], dim=-1).to(solver.ftype)

    sigmas = solver.gather_nonpiston(scales).numpy(force=True)
    covariance = np.diag(sigmas ** 2)

    if include_jitter and solver.z_jitter > 0:
        direction = defocus_direction(solver).numpy(force=True)
        covariance = covariance + _uniform_z_jitter_variance(solver) * np.outer(direction, direction)

    return covariance


def augmented_prior_covariance(solver: ZstackSolver) -> np.ndarray:
    """
    Extends the output of `coefficient_prior_covariance`, adding z along the last row+column.
    Returns as a [C_kept+1, C_kept+1] covariance matrix matching `calculate_augmented_fisher_matrix()`.
    """
    if solver.z_jitter <= 0:
        raise ValueError("z_jitter = 0, so z is exactly known, use `coefficient_prior_covariance`")

    num_coefs = solver.num_nonpiston_coefs
    covariance = np.zeros((num_coefs + 1, num_coefs + 1))
    covariance[:num_coefs, :num_coefs] = coefficient_prior_covariance(solver, include_jitter=False)
    covariance[num_coefs, num_coefs] = _uniform_z_jitter_variance(solver)
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
