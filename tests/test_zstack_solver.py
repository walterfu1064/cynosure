"""
Unit tests for zstack_solver.py.
"""

import re

import pytest
import torch

from cynosure.config import (
    BlobPriorConfig,
    SimulationConfig,
    OpticalConfig,
    PriorConfig,
    TrainingConfig,
    ZernikeConfig,
)
from cynosure.config_defaults import get_default_mixture_config, get_default_training_config
from cynosure.object_distribution import FixedBead, SampledKBlobs
from cynosure.zstack_decoding.jitter_modes import ShellJitter, SoftShellJitter, UniformJitter
from cynosure.zstack_decoding.posterior_heads import CnnEncoderSpec
from cynosure.zstack_decoding.zstack_solver import (
    ZstackSolver_MLE,
    ZstackSolver_MixedDensity,
)


def make_train_cfg(**overrides) -> TrainingConfig:
    """Training config for tests: small generator chunks, everything else default"""
    return get_default_training_config(generator_chunk=8, **overrides)


def make_solver(
        *,
        z_objective=None,
        z_jitter=0.0,
        phase_allowed=((2, 0),),
        amp_allowed=((0, 0),),
        solver_cls=ZstackSolver_MLE,
        train_cfg=None,
        object_distrib_factory=None,
        **kwargs,
):
    """
    Builds a minimal solver for testing synthetic data generation.

    Object is a small FixedBead by default, unless a factory to building a
    different variant is passed as `object_distrib_factory`.
    """
    sim_cfg = SimulationConfig(
        pupil_grid_size=63,
        object_grid_size=31,
        object_pixel_size=0.1,
        ftype=torch.float32,
        ctype=torch.complex64,
)
    opt_cfg = OpticalConfig(
        wavelength=0.51,
        focal_length=4.5e3,
        numerical_aperture=0.9,
        aperture_type="flat",
        medium_index=1.54,
        immersion_index=1.0,  # deliberate mismatch, so tests include axial scaling confirmation
    )
    phase_cfg = ZernikeConfig(max(n for n, _ in phase_allowed), allowed_nm=phase_allowed)
    amp_cfg = ZernikeConfig(max(n for n, _ in amp_allowed), allowed_nm=amp_allowed)  # piston only, by default
    if object_distrib_factory is None:
        object_distrib = FixedBead(sim_cfg, 0.1)
    else:
        object_distrib = object_distrib_factory(sim_cfg)
    encoder_spec = CnnEncoderSpec(
        spatial_hidden_channels=(16, 32),
        embedding_dims=128,
    )

    return solver_cls(
        train_cfg=train_cfg if train_cfg is not None else make_train_cfg(),
        sim_cfg=sim_cfg,
        optics_cfg=opt_cfg,
        phase_cfg=phase_cfg,
        amp_cfg=amp_cfg,
        phase_prior_cfg=PriorConfig(0.05, 1),
        amp_prior_cfg=PriorConfig(0.05, 2),
        object_distribution=object_distrib,
        z_objective=z_objective,
        z_jitter=z_jitter,
        encoder_spec=encoder_spec,
        **kwargs,
    )


def make_blob_solver(num_blobs: int = 1, **kwargs) -> ZstackSolver_MLE:
    """
    Builds a solver with a stochastic SampledKBlobs object for testing join data + object generation.
    """
    blob_cfg = BlobPriorConfig(
        position_sigma=0.5,
        reference_diameter=0.5,
        log_diameter_sigma=0.3,
        amplitude_logit_sigma=2.0,
    )
    return make_solver(
        object_distrib_factory=lambda sim_cfg: SampledKBlobs(sim_cfg, num_blobs, blob_cfg),
        **kwargs,
    )


def _rel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L1 distance between two tensors"""
    return ((a - b).abs().sum() / b.abs().sum()).item()


def test_z_jitter_to_phase_conversion():
    """Unaberrated image with z-offset ~= defocus-aberrated image at nominal focal plane"""
    solver = make_solver(z_objective=torch.zeros(1), z_jitter=1.0, phase_allowed=((2, 0), (4, 0), (6, 0)))
    offsets = torch.tensor([0.8], dtype=solver.ftype)
    zero_phase = torch.zeros(1, solver.coefficients.block_size("phase"), dtype=solver.ftype)  # no phase aberrations
    amp = solver.coefficients.is_amp_pinned.to(solver.ftype).unsqueeze(0)  # no amp aberrations

    img_offset = solver.simulator.simulate_normalized_stacks(zero_phase, amp, offsets=offsets)  # z-jittered, unaberrated
    defocus_phase = zero_phase + solver.simulator.z_offset_to_coefficients(offsets)

    img_label = solver.simulator.simulate_normalized_stacks(defocus_phase, amp, offsets=None)  # nominal z, defocus-aberrated
    assert _rel_l1(img_label, img_offset) < 5e-3  # should agree

    img_unfolded = solver.simulator.simulate_normalized_stacks(zero_phase, amp, offsets=None)  # nominal z, unaberrated
    assert _rel_l1(img_unfolded, img_offset) > 0.05  # should disagree


def test_reconstruction_through_z_offset():
    """Aberrated labels at a z-offset can be reconstructed, substituting z-offset for aberrations"""
    solver = make_solver(z_jitter=2.0, phase_allowed=((2, 0), (4, 0), (6, 0)))
    generator = torch.Generator().manual_seed(0)
    images, _, phase_coefs, amp_coefs = solver.simulator.create_examples(8, generator=generator)

    recon = solver.simulator.simulate_normalized_stacks(phase_coefs, amp_coefs, offsets=None)
    assert _rel_l1(recon, images) < 5e-3


def _mixing_logit_grads(solver: ZstackSolver_MixedDensity) -> torch.Tensor:
    """Runs one loss backward and returns the gradient reaching the mixing logits"""
    generator = torch.Generator().manual_seed(0)
    images, z, phase_coefs, amp_coefs = solver.simulator.create_examples(4, generator=generator)
    targets = solver.coefficients.whiten_blocks(phase_coefs, amp_coefs)

    predictions = solver.forward(images, z).detach().requires_grad_(True)
    loss, _ = solver.compute_losses(predictions, targets)
    loss.backward()

    num_means = solver.num_components * solver.coefficients.num_coefs
    return predictions.grad[:, num_means:num_means + solver.num_components]


def test_mixing_warmup_freezes_weights():
    """During the mixing warmup the logits get no gradient; without it (or after it) they do"""
    kwargs = dict(z_jitter=1.0, phase_allowed=((2, 0), (2, 2)), solver_cls=ZstackSolver_MixedDensity)

    # current_epoch is 0 without a trainer, so a warmup of 1 epoch is still in effect
    warm = make_solver(mixture_cfg=get_default_mixture_config(mixing_warmup_epochs=1), **kwargs)
    assert torch.all(_mixing_logit_grads(warm) == 0)

    cold = make_solver(mixture_cfg=get_default_mixture_config(mixing_warmup_epochs=0), **kwargs)
    assert torch.any(_mixing_logit_grads(cold) != 0)


def test_deterministic_label_seeding():
    """Synthesized coefficients and object parameters are fully reproducible when seeded"""
    solver = make_blob_solver(num_blobs=2)
    outputs = solver.simulator.create_examples(512, generator=torch.Generator().manual_seed(0))
    repeats = solver.simulator.create_examples(512, generator=torch.Generator().manual_seed(0))
    for a, b in zip(outputs, repeats):  # seeding correctly makes everything fully deterministic
        torch.testing.assert_close(a, b)


def test_coefficient_blocks():
    """With a stochastic object, solver generates phase, amp, and object CoefficientBlocks that whiten to N(0, 1)"""
    solver = make_blob_solver(num_blobs=2, phase_allowed=((2, 0), (2, 2), (3, 1)), amp_allowed=((0, 0), (1, 1), (2, 0)))
    coefs = solver.coefficients
    assert [block.name for block in coefs.blocks] == ["phase", "amp", "object"]
    assert coefs.block_sizes == (3, 3, solver.object_distribution.num_params) == (3, 3, 7)

    outputs = solver.simulator.create_examples(512, generator=torch.Generator().manual_seed(0))
    assert len(outputs) == 3 + 2  # images and nominal z, plus one label tensor per block

    targets = coefs.whiten_blocks(*outputs[2:])
    assert targets.shape == (512, coefs.num_coefs)
    for name in ("phase", "amp", "object"):  # every block whitens to ~N(0, 1), excluding pinned coefs
        whitened = coefs.block_coefs(targets, name)
        pinned = coefs.block_coefs(coefs.is_pinned, name)
        assert torch.all(whitened[:, pinned] == 0), f"{name} pinned coefs should whiten to exactly zero"
        free = whitened[:, ~pinned]
        assert free.mean(dim=0).abs().max() < 0.2, f"{name} labels are off-center"  # object needs empirical means
        assert (free.std(dim=0) - 1).abs().max() < 0.2, f"{name} labels are mis-scaled"

    assert len(coefs.sample(4)) == 2  # `CoefficientSpace.sample` shouldn't draw the object block itself


@pytest.mark.parametrize("jitter", [UniformJitter(1.0), ShellJitter(0.25, 1.0), SoftShellJitter(1.0)])
def test_jitter_widened_whitening(jitter):
    """Jittered, defocus-coupled phase labels still whiten to ~N(0, 1)"""
    solver = make_solver(z_jitter=jitter, phase_allowed=((2, 0), (4, 0), (2, 2)))
    coefs = solver.coefficients
    axial_scale = solver.simulator.propagator.axial_scale
    assert coefs.jitter_variance == pytest.approx(jitter.variance * axial_scale ** 2)

    _, _, phase_coefs, amp_coefs = solver.simulator.create_examples(1024, generator=torch.Generator().manual_seed(0))
    whitened = coefs.block_coefs(coefs.whiten_blocks(phase_coefs, amp_coefs), "phase")
    assert whitened.mean(dim=0).abs().max() < 0.15
    assert (whitened.std(dim=0) - 1).abs().max() < 0.1

    prior_scales = coefs.block_coefs(coefs.prior_scales, "phase")
    target_scales = coefs.block_coefs(coefs.target_scales, "phase")
    assert (phase_coefs[:, 0] / prior_scales[0]).std() > 5  # folded offsets swamp the defocus coef's prior
    assert target_scales[0] / prior_scales[0] > 5  # folded offsets swap the defocus coef's prior
    assert target_scales[1] == pytest.approx(prior_scales[1])  # uncoupled coefficients are left alone


def test_object_labels_describe_imaged_object():
    """Synthesized coef + object labels correctly reproduce the synthesized images"""
    solver = make_blob_solver(num_blobs=2)
    generator = torch.Generator().manual_seed(0)
    images, _, phase_coefs, amp_coefs, object_params = solver.simulator.create_examples(8, generator=generator)

    objects = solver.object_distribution.render_from_params(object_params)
    recon = solver.simulator.simulate_normalized_stacks(phase_coefs, amp_coefs, objects=objects)
    assert _rel_l1(recon, images) < 5e-3


def test_twin_sign_parity():
    """Phase coefficients flip on even n, amp on odd n"""
    solver = make_blob_solver(
        z_objective=torch.zeros(1),
        z_jitter=1.0,
        phase_allowed=((2, 0), (2, 2), (3, 1), (3, 3), (4, 2)),
        twin_augmentation=True,
    )
    for label, sign in zip(solver.coefficients.blocks[0].labels, solver.twin_phase_signs.tolist()):
        n = int(re.search(r"_\{(\d+)\}", label).group(1))
        assert sign == (1.0 if n % 2 else -1.0), f"wrong twin sign for phase {label}"
    assert solver.twin_amp_signs.tolist() == [1.0]  # amp piston, n = 0, twin-invariant


def test_twin_labels_produce_identical_images():
    """Image is invariant across the twin transform (negate z and specific coefficients)"""
    solver = make_solver(
        z_objective=torch.zeros(1),
        z_jitter=1.0,
        phase_allowed=((2, 0), (2, 2), (3, 1), (3, 3), (4, 0)),
        twin_augmentation=True,
    )
    sim = solver.simulator
    generator = torch.Generator().manual_seed(0)
    phase_coefs, amp_coefs = sim.sample_coefficients(4, generator=generator)
    phase_coefs = phase_coefs * 10  # exaggerate so asymmetries are clear
    offsets = torch.tensor([0.9, -0.5, 0.3, 0.0], dtype=solver.ftype)

    images = sim.simulate_normalized_stacks(phase_coefs, amp_coefs, offsets=offsets)
    twins = sim.simulate_normalized_stacks(
        phase_coefs * solver.twin_phase_signs, amp_coefs * solver.twin_amp_signs, offsets=-offsets,
    )
    assert _rel_l1(twins, images) < 1e-4  # identical images from twin transform

    mirrored_only = sim.simulate_normalized_stacks(phase_coefs, amp_coefs, offsets=-offsets)
    assert _rel_l1(mirrored_only, images) > 0.05  # distinct images from mirroring z alone


def test_twin_augmentation_doubles_training_batches():
    """The augmented step doubles training rows with twin labels, but leaves eval mode alone"""
    solver = make_blob_solver(
        z_objective=torch.zeros(1),
        z_jitter=1.0,
        phase_allowed=((2, 0), (3, 1)),
        train_cfg=make_train_cfg(batch_size=4),
        twin_augmentation=True,
    )
    assert solver.training
    _, predictions, targets, _ = solver._forwards_common(generator=torch.Generator().manual_seed(0))
    assert predictions.shape[0] == targets.shape[0] == 8

    phase, amp, objects = solver.coefficients.unwhiten_to_blocks(targets[:4])
    twin_phase, twin_amp, twin_objects = solver.coefficients.unwhiten_to_blocks(targets[4:])
    torch.testing.assert_close(twin_phase, phase * solver.twin_phase_signs)
    torch.testing.assert_close(twin_amp, amp * solver.twin_amp_signs)
    torch.testing.assert_close(twin_objects, objects)  # object is twin-invariant

    solver = solver.eval()
    _, predictions, targets, _ = solver._forwards_common(generator=torch.Generator().manual_seed(0))
    assert predictions.shape[0] == targets.shape[0] == 4
