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


def calculate_fisher_matrix(
        coefs: torch.Tensor,
        z: torch.Tensor,
        solver: ZstackSolver,
        photons: Optional[float] = None,
        background: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fisher matrix of a simulated z-stack with respect to the non-piston aberration coefficients.

    Arguments:
    - coefs: [C_kept,], the free (non-piston) coefficients, as from `ZstackSolver.gather_nonpiston()`
    - z: [Z,]
    - photons, background: scalars, use NoiseCfg averages if None
    Returns:
    - simulated z-stack, [Z, H, W]
    - Fisher matrix, [Z, C_kept, C_kept]
    """
    if solver.noise_cfg:
        noise_defaults = {
            "photons": solver.noise_cfg.average_photons,
            "background": solver.noise_cfg.average_background,
            "read": solver.noise_cfg.read_noise,
        }
    else:
        noise_defaults = {
            "photons": 1,
            "background": 0,
            "read": 0,
        }
    photons = photons or noise_defaults["photons"]
    background = background or noise_defaults["background"]
    read_noise = noise_defaults["read"]

    counts = simulate_image_as_photon_counts(coefs, z, photons, background, solver)
    jac = calculate_image_jacobian(coefs, z, photons, background, solver)
    inv_variance = 1.0 / (counts + read_noise ** 2)  # [Z, H, W]
    fisher_matrix = torch.einsum("zhw,zhwi,zhwj->zij", inv_variance, jac, jac)  # [Z, C_kept, C_kept]
    return counts, fisher_matrix


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
