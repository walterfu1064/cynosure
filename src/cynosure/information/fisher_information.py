from typing import Optional

import torch

from cynosure.zstack_decoding import ZstackSolver
from cynosure.utilities.fft_utilities import convolve_psf_with_object


def simulate_image_as_photon_counts(
        coefs: torch.Tensor,
        z: torch.Tensor,
        num_photons: float,
        background_level: float,
        solver: ZstackSolver,
) -> torch.Tensor:
    """
    Forwards-simulates a z-stack and normalizes it to raw photon counts.

    Arguments:
    - coefs: [C,], phase and amplitude together, as from `ZstackSolver.join_phase_amp()`
    - z: [Z,]
    - num_photons, background_level: scalars
    - object_distrib: [H, W]
    Returns:
    - [Z, H, W]
    """
    num_z = z.shape[0]

    defocus = solver.propagator.defocus_from_objective_z(z)
    phase_coefs, amp_coefs = solver.split_phase_amp(coefs)
    phase_coefs = phase_coefs.unsqueeze(0).expand(num_z, -1)
    amp_coefs = amp_coefs.unsqueeze(0).expand(num_z, -1)
    psf = solver.propagator(defocus, phase_coefs, amp_coefs)
    images = convolve_psf_with_object(solver.object_distribution, psf)
    images.clamp_(min=0)

    photon_counts = torch.full((num_z, 1, 1), num_photons, device=solver.device, dtype=images.dtype)
    background = torch.full((num_z, 1, 1), background_level, device=solver.device, dtype=images.dtype)
    return (images / images.sum((-2, -1), keepdim=True)) * photon_counts + background


def calculate_image_jacobian(
        coefs: torch.Tensor,
        z: torch.Tensor,
        num_photons: float,
        background_level: float,
        solver: ZstackSolver,
) -> torch.Tensor:
    """
    Jacobian of the simulated z-stack with respect to each aberration coefficient.

    Arguments:
    - coefs: [C,], phase and amplitude together, as from `ZstackSolver.join_phase_amp()`
    - z: [Z,]
    - num_photons, background_level: scalars
    - object_distrib: [H, W]
    Returns:
    - [Z, H, W, C]
    """
    jac = torch.func.jacfwd(simulate_image_as_photon_counts)(coefs, z, num_photons, background_level, solver)
    return jac


def calculate_fisher_matrix(
        coefs: torch.Tensor,
        z: torch.Tensor,
        solver: ZstackSolver,
        num_photons: Optional[float] = None,
        background_level: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fisher matrix of the simulated z-stack with respect to the aberration coefficients.

    Arguments:
    - coefs: [C,], phase and amplitude together, as from `ZstackSolver.join_phase_amp()`
    - z: [Z,]
    - num_photons, background_level: scalars, use NoiseCfg averages if None
    - object_distrib: [H, W]
    Returns:
    - simulated z-stack, [Z, H, W]
    - Fisher matrix, [Z, C, C]
    """
    if num_photons is None:
        num_photons = (solver.noise_cfg.max_photons + solver.noise_cfg.min_photons) / 2
    if background_level is None:
        background_level = solver.noise_cfg.max_background / 2
    counts = simulate_image_as_photon_counts(coefs, z, num_photons, background_level, solver)
    jac = calculate_image_jacobian(coefs, z, num_photons, background_level, solver)
    inv_variance = 1.0 / (counts + solver.noise_cfg.read_noise ** 2)  # [Z, H, W]
    fisher_matrix = torch.einsum("zhw,zhwi,zhwj->zij", inv_variance, jac, jac)  # [Z, C, C]
    return counts, fisher_matrix
