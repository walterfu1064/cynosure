"""
Unit tests for zstack_solver.py.
"""

import torch

from cynosure.config import SimulationConfig, OpticalConfig, PriorConfig, ZernikeConfig
from cynosure.zstack_decoding.zstack_solver import ZstackSolver, make_object_distribution


def make_solver(*, z_objective=None, z_jitter=0.0, phase_allowed=((2, 0),)):
    """Builds a minimal solver for testing synthetic data generation"""
    sim_cfg = SimulationConfig(pupil_grid_size=63, object_grid_size=31, object_pixel_size=0.1)
    opt_cfg = OpticalConfig(
        wavelength=0.51,
        focal_length=4.5e3,
        numerical_aperture=0.9,
        aperture_type="flat",
        medium_index=1.54,
        immersion_index=1.0,  # deliberate mismatch, so tests include axial scaling confirmation
    )
    max_n = max(n for n, _ in phase_allowed)
    phase_cfg = ZernikeConfig(max_n, allowed_nm=phase_allowed)
    amp_cfg = ZernikeConfig(0)  # amplitude piston only
    object_distrib = make_object_distribution(sim_cfg.object_grid_size, sim_cfg.object_pixel_size, 0.1)

    return ZstackSolver(
        sim_cfg=sim_cfg,
        optics_cfg=opt_cfg,
        phase_cfg=phase_cfg,
        amp_cfg=amp_cfg,
        phase_prior_cfg=PriorConfig(0.05, 1),
        amp_prior_cfg=PriorConfig(0.05, 2),
        object_distribution=object_distrib,
        z_objective=z_objective,
        z_jitter=z_jitter,
        generator_chunk=8,
    )


def _rel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L1 distance between two tensors"""
    return ((a - b).abs().sum() / b.abs().sum()).item()


def test_z_jitter_to_phase_conversion():
    """Unaberrated image with z-offset ~= defocus-aberrated image at nominal focal plane"""
    solver = make_solver(z_objective=torch.zeros(1), z_jitter=1.0, phase_allowed=((2, 0), (4, 0), (6, 0)))
    offsets = torch.tensor([0.8], dtype=solver.ftype)
    zero_phase = torch.zeros(1, solver.num_phase_coefs, dtype=solver.ftype)  # no phase aberrations
    amp = solver.is_amp_piston.to(solver.ftype).unsqueeze(0)  # no amp aberrations

    img_offset = solver.simulate_normalized_stacks(zero_phase, amp, offsets=offsets)  # z-jittered, unaberrated
    defocus_phase = zero_phase + solver._z_jitter_to_phase_coefs(offsets)

    img_label = solver.simulate_normalized_stacks(defocus_phase, amp, offsets=None)  # nominal z, defocus-aberrated
    assert _rel_l1(img_label, img_offset) < 5e-3  # should agree

    img_unfolded = solver.simulate_normalized_stacks(zero_phase, amp, offsets=None)  # nominal z, unaberrated
    assert _rel_l1(img_unfolded, img_offset) > 0.05  # should disagree


def test_reconstruction_through_z_offset():
    """Aberrated labels at a z-offset can be reconstructed, substituting z-offset for aberrations"""
    solver = make_solver(z_jitter=2.0, phase_allowed=((2, 0), (4, 0), (6, 0)))
    generator = torch.Generator().manual_seed(0)
    images, phase_coefs, amp_coefs = solver.create_examples(8, generator=generator)

    recon = solver.simulate_normalized_stacks(phase_coefs, amp_coefs, offsets=None)
    assert _rel_l1(recon, images) < 5e-3
