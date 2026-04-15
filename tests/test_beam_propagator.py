"""
Unit tests for beam_propagator.py.
"""

import math

import numpy as np
import pytest
import torch
from scipy.special import j1

from beam_propagation.beam_propagator import (
    BeamPropagator, SimulationConfig, OpticalConfig, ZernikeConfig,
)


def params():
    """Returns handy defaults, all distances in um"""
    return dict(
        wavelength=0.5,
        focal_length=5e3,
        NA=0.2,
        pupil_grid_size=255,
        object_grid_size=255,
        max_n=4,
        ftype=torch.float64,
        ctype=torch.complex128,
    )


def make_propagator(
        *,
        NA=0.2,
        wavelength=0.5,
        focal_length=5e3,
        pupil_grid_size=255,
        object_grid_size=255,
        samples_per_airy_radius=10.0,
        max_n=4,
        ftype=torch.float64,
        ctype=torch.complex128,
):
    """Builds a low-NA beam propagator for testing against an Airy beam"""
    airy_radius = 1.22 * wavelength / NA
    object_pixel_size = airy_radius / samples_per_airy_radius
    sim = SimulationConfig(
        pupil_grid_size=pupil_grid_size,
        object_grid_size=object_grid_size,
        object_pixel_size=object_pixel_size,
    )
    opt = OpticalConfig(wavelength=wavelength, focal_length=focal_length, numerical_aperture=NA)
    zer = ZernikeConfig(max_n=max_n)
    return BeamPropagator(sim, opt, zer, ftype=ftype, ctype=ctype)


def zero_ab(prop, batch=1):
    """Returns a tensor of zero-aberration coefficients"""
    return torch.zeros(batch, prop.num_aberrations, dtype=prop.ftype)


def analytic_airy(r_abs, wavelength, NA):
    k = 2 * math.pi / wavelength
    v = k * NA * r_abs
    v_safe = np.where(v != 0, v, 1.0)
    return np.where(v != 0, (2 * j1(v_safe) / v_safe) ** 2, 1.0)


def parabolic_subpixel_peak(img):
    """Return (iy, ix) with sub-pixel refinement via parabolic interpolation"""
    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    dy = dx = 0.0
    if 0 < ix < img.shape[1] - 1:
        a, b, c = img[iy, ix - 1], img[iy, ix], img[iy, ix + 1]
        denom = a - 2 * b + c
        if denom != 0:
            dx = 0.5 * (a - c) / denom
    if 0 < iy < img.shape[0] - 1:
        a, b, c = img[iy - 1, ix], img[iy, ix], img[iy + 1, ix]
        denom = a - 2 * b + c
        if denom != 0:
            dy = 0.5 * (a - c) / denom
    return iy + dy, ix + dx


class TestFocalIntensity:
    def test_airy_profile(self):
        """Radial profile at z=0 must match the analytic Airy pattern"""
        p = params()
        prop = make_propagator(**p)
        with torch.no_grad():
            I = prop(zero_ab(prop), torch.zeros(1, dtype=prop.ftype))[0].cpu().numpy()

        cy = prop.object_grid_size // 2
        profile = I[cy]
        x = prop.object_x[cy].cpu().numpy()
        profile_n = profile / profile.max()
        airy = analytic_airy(np.abs(x), p["wavelength"], p["NA"])

        rmse = float(np.sqrt(np.mean((profile_n - airy) ** 2)))
        assert rmse < 5e-3

    def test_focus_centered(self):
        """Peak lands on the central pixel for zero aberrations at z=0"""
        prop = make_propagator()
        with torch.no_grad():
            I = prop(zero_ab(prop), torch.zeros(1, dtype=prop.ftype))[0].cpu().numpy()
        iy, ix = np.unravel_index(np.argmax(I), I.shape)
        c = prop.object_grid_size // 2
        assert (iy, ix) == (c, c)

    def test_marechal_strehl(self):
        """Small astigmatism gives Strehl matching the Marechal approximation"""
        prop = make_propagator()
        z0 = torch.zeros(1, dtype=prop.ftype)
        with torch.no_grad():
            I0_peak = prop(zero_ab(prop), z0)[0].max().item()

        a = 0.05  # waves RMS
        ab = zero_ab(prop)
        ab[0, 5] = a  # bank index 5 = Noll 6 = astigmatism (n=2, m=2)
        with torch.no_grad():
            I_ab_peak = prop(ab, z0)[0].max().item()

        strehl = I_ab_peak / I0_peak
        marechal = 1.0 - (2 * math.pi * a) ** 2
        assert strehl == pytest.approx(marechal, rel=0.03)


class TestSymmetry:
    def test_rotational_symmetry(self):
        """Clear circular pupil produces a PSF invariant under 90-degree rotation"""
        prop = make_propagator()
        with torch.no_grad():
            I = prop(zero_ab(prop), torch.zeros(1, dtype=prop.ftype))[0].cpu().numpy()
        assert np.allclose(I, np.rot90(I), atol=1e-10 * I.max())

    def test_z_reversal_symmetry(self):
        """I(x, y, +z) == I(x, y, -z) for a symmetric aberration-free pupil"""
        prop = make_propagator()
        z = torch.tensor([3e-6, -3e-6], dtype=prop.ftype)
        with torch.no_grad():
            I = prop(zero_ab(prop, batch=2), z).cpu().numpy()
        assert np.allclose(I[0], I[1], atol=1e-10 * I[0].max())


class TestAberrations:
    def test_tilt_shifts_focus(self):
        """x-tilt of a waves shifts the focal spot by 2*a*lambda/NA along x"""
        p = params()
        a_tilt = 2.0
        expected_shift = 2.0 * a_tilt * p["wavelength"] / p["NA"]

        prop = make_propagator(
            **p | dict(
                samples_per_airy_radius=20.0,
                pupil_grid_size=511,
                object_grid_size=511,
            ),
        )
        ab = zero_ab(prop)
        ab[0, 1] = a_tilt  # bank index 1 = Noll 2 = x-tilt
        with torch.no_grad():
            I = prop(ab, torch.zeros(1, dtype=prop.ftype))[0].cpu().numpy()

        iy, ix = parabolic_subpixel_peak(I)
        c = prop.object_grid_size // 2
        measured_shift_x = (ix - c) * prop.object_dx
        measured_shift_y = abs(iy - c) * prop.object_dx

        assert measured_shift_y < 0.5 * prop.object_dx
        assert measured_shift_x == pytest.approx(expected_shift, rel=0.01)

    def test_axial_sinc_profile(self):
        """On-axis intensity vs z matches the paraxial sinc^2 prediction"""
        NA, wl = 0.1, 0.5
        prop = make_propagator(
            NA=NA, wavelength=wl,
            pupil_grid_size=255, object_grid_size=65,
            samples_per_airy_radius=6.0,
        )
        first_zero = 2 * wl / NA ** 2
        zs = torch.linspace(-0.6 * first_zero, 0.6 * first_zero, 13, dtype=prop.ftype)
        B = zs.numel()

        with torch.no_grad():
            I = prop(zero_ab(prop, batch=B), zs).cpu().numpy()
        c = prop.object_grid_size // 2
        on_axis = I[:, c, c]
        on_axis_n = on_axis / on_axis.max()

        x = zs.cpu().numpy() * NA ** 2 / (2 * wl)
        analytic = np.sinc(x) ** 2

        rmse = float(np.sqrt(np.mean((on_axis_n - analytic) ** 2)))
        assert rmse < 5e-3


class TestInfrastructure:
    def test_batch_consistency(self):
        """Batched forward pass must equal serial single-item forward passes"""
        prop = make_propagator()
        B = 4
        torch.manual_seed(0)
        ab = 0.05 * torch.randn(B, prop.zernike_bank.shape[0], dtype=prop.ftype)
        z = torch.linspace(-1, 1, B, dtype=prop.ftype)

        with torch.no_grad():
            I_batch = prop(ab, z)
            I_serial = torch.stack([prop(ab[i:i + 1], z[i:i + 1])[0] for i in range(B)])

        # Batched FFT routines can accumulate differently; demand relative agreement < 1e-6.
        assert torch.allclose(I_batch, I_serial, atol=1e-6 * I_batch.abs().max())

    def test_differentiable(self):
        """Autograd should flow through the full forward pass and produce finite gradients"""
        prop = make_propagator(pupil_grid_size=127, object_grid_size=127)
        N = prop.zernike_bank.shape[0]
        ab = torch.full((1, N), 0.05, dtype=prop.ftype, requires_grad=True)
        z = torch.zeros(1, dtype=prop.ftype)

        I = prop(ab, z)
        I.sum().backward()

        assert ab.grad is not None
        assert ab.grad.shape == ab.shape
        assert torch.isfinite(ab.grad).all()
        assert ab.grad.abs().sum() > 0
