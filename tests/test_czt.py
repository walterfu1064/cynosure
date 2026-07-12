"""
Unit tests for czt1d.py.
"""

import numpy as np
import pytest
import torch

from wave_physics.czt.czt1d import ChirpZTransform1D
from wave_physics.czt.czt2d import ChirpZTransform2D


def params_1d():
    npts, mpts = 128, 256
    dx = 0.02
    ftype, ctype = torch.float64, torch.complex128
    return npts, mpts, dx, ftype, ctype


def make_signal_1d(x: torch.Tensor) -> torch.Tensor:
    f1, gamma1 = 3.0, 2.0
    f2 = 4.0
    y1 = torch.sin(2 * torch.pi * f1 * x) * torch.exp(-f1 * x / gamma1)
    y2 = 0.2 * torch.sin(2 * torch.pi * f2 * x)
    return y1 + y2


def make_signal_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    X, Y = torch.meshgrid(x, y, indexing="xy")
    fx1, fy1 = 2.5, 1.3
    fx2, fy2 = 4.1, 3.7
    s1 = torch.sin(2 * torch.pi * (fx1 * X + fy1 * Y))
    s2 = 0.3 * torch.cos(2 * torch.pi * (fx2 * X - fy2 * Y))
    envelope = torch.exp(-0.5 * (X**2 + Y**2) / 4.0)
    return (s1 + s2) * envelope


def fft_bin_aligned_range(npts: int, dx: float, k_start: int, mpts: int) -> tuple[float, float]:
    """Returns (f_start, f_end) such that mpts CZT samples land exactly on FFT bins
    k_start, k_start+1, ..., k_start+mpts-1."""
    bin_spacing = 1.0 / (npts * dx)
    f_start = k_start * bin_spacing
    f_end = (k_start + mpts - 1) * bin_spacing
    return f_start, f_end


class TestCZT1D:
    def test_freq_grid(self):
        """CZT-sampled frequency grid should linearly span the proscribed range"""
        npts, mpts, dx, ftype, ctype = params_1d()
        f_start, f_end = 2.0, 6.0
        czt = ChirpZTransform1D(npts, mpts, dx, f_start, f_end, ftype=ftype, ctype=ctype)

        assert czt.freq.shape == (mpts,)
        assert czt.freq.dtype == ftype
        assert czt.freq[0].item() == pytest.approx(f_start)
        assert czt.freq[-1].item() == pytest.approx(f_end)
        diffs = torch.diff(czt.freq)
        assert torch.allclose(diffs, diffs[0] * torch.ones_like(diffs))

    def test_fft_agreement(self):
        """CZT should agree with FFT at the sampled points"""
        npts, mpts, dx, ftype, ctype = params_1d()
        n = torch.arange(npts, dtype=ftype)
        y = make_signal_1d(n * dx)

        f = torch.fft.fftshift(torch.fft.fftfreq(npts, d=dx)).numpy()
        yft = torch.fft.fftshift(torch.fft.fft(y)).numpy()
        yft = np.abs(yft)

        f_start, f_end = 2.0, 6.0
        czt = ChirpZTransform1D(npts, mpts, dx, f_start, f_end, ftype=ftype, ctype=ctype)
        fczt = czt.freq.numpy()
        with torch.no_grad():
            yczt = czt(y).numpy()
        yczt = np.abs(yczt)

        win = (f >= f_start) & (f <= f_end)
        assert np.all(np.diff(fczt) > 0)  # really a check on np.interp, not on czt
        yczt_interp = np.interp(f[win], fczt, yczt)
        assert np.allclose(yft[win], yczt_interp, atol=1e-2)

    def test_single_tone_peak_location(self):
        """CZT of a pure tone should show a peak at the fundamental"""
        npts, mpts, dx, ftype, ctype = params_1d()
        f_tone = 4.25  # deliberately off the FFT grid
        n = torch.arange(npts, dtype=ftype)
        y = torch.cos(2 * torch.pi * f_tone * n * dx)

        f_start, f_end = 3.0, 6.0
        czt = ChirpZTransform1D(npts, mpts, dx, f_start, f_end, ftype=ftype, ctype=ctype)
        with torch.no_grad():
            yczt = czt(y)

        peak_idx = torch.argmax(torch.abs(yczt)).item()
        peak_freq = czt.freq[peak_idx].item()
        df = (f_end - f_start) / (mpts - 1)
        assert abs(peak_freq - f_tone) <= df

    def test_batched_input(self):
        """CZT should operate along the last dim regardless of leading batch dims"""
        npts, mpts, dx, ftype, ctype = params_1d()
        n = torch.arange(npts, dtype=ftype)
        x = n * dx
        y0 = make_signal_1d(x)
        y1 = torch.cos(2 * torch.pi * 3.7 * x)
        y2 = make_signal_1d(x + 0.1)
        batch = torch.stack([y0, y1, y2]).reshape(3, 1, npts)

        f_start, f_end = 2.0, 6.0
        czt = ChirpZTransform1D(npts, mpts, dx, f_start, f_end, ftype=ftype, ctype=ctype)
        with torch.no_grad():
            out_batch = czt(batch)
            out_0 = czt(y0)
            out_1 = czt(y1)
            out_2 = czt(y2)

        assert out_batch.shape == (3, 1, mpts)
        assert torch.allclose(out_batch[0, 0], out_0)
        assert torch.allclose(out_batch[1, 0], out_1)
        assert torch.allclose(out_batch[2, 0], out_2)

    def test_linearity(self):
        """CZT should be linear"""
        npts, mpts, dx, ftype, ctype = params_1d()
        n = torch.arange(npts, dtype=ftype)
        x = n * dx
        y_a = make_signal_1d(x)
        y_b = torch.cos(2 * torch.pi * 3.7 * x)
        a, b = 1.7, -0.4

        czt = ChirpZTransform1D(npts, mpts, dx, 2.0, 6.0, ftype=ftype, ctype=ctype)
        with torch.no_grad():
            lhs = czt(a * y_a + b * y_b)
            rhs = a * czt(y_a) + b * czt(y_b)
        assert torch.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)

    def test_differentiable(self):
        """CZT should support gradient flow"""
        npts, mpts, dx, ftype, ctype = params_1d()
        n = torch.arange(npts, dtype=ftype)
        y = make_signal_1d(n * dx).clone().requires_grad_(True)

        czt = ChirpZTransform1D(npts, mpts, dx, 2.0, 6.0, ftype=ftype, ctype=ctype)
        out = czt(y)
        loss = (out.abs() ** 2).sum()
        loss.backward()

        assert y.grad is not None
        assert y.grad.shape == y.shape
        assert torch.isfinite(y.grad).all()
        assert y.grad.abs().sum() > 0


class TestCZT2D:
    def test_agreement_with_fft2_on_bins(self):
        """CZT2D on FFT bins should agree with fft2"""
        npts = 64
        mpts = 32
        dx = 0.02
        ftype, ctype = torch.float64, torch.complex128

        n = torch.arange(npts, dtype=ftype)
        coord = n * dx
        signal = make_signal_2d(coord, coord)

        y_fft2 = torch.fft.fft2(signal)

        k_start = 5
        f_start, f_end = fft_bin_aligned_range(npts, dx, k_start, mpts)

        czt = ChirpZTransform2D(
            num_input_points=npts,
            num_output_points=mpts,
            input_step=dx,
            start_frequency=f_start,
            end_frequency=f_end,
            ftype=ftype,
            ctype=ctype,
        )
        with torch.no_grad():
            out = czt(signal)

        expected = y_fft2[k_start:k_start + mpts, k_start:k_start + mpts]
        assert torch.allclose(out, expected, atol=1e-10, rtol=1e-10)

    def test_asymmetric_axis_mapping(self):
        """CZT2D should operate in dim order (penultimate, last), and will show errors for asymmetric axes"""
        npts_y, npts_x = 48, 80
        mpts_y, mpts_x = 24, 40
        dx_y, dx_x = 0.03, 0.02
        ftype, ctype = torch.float64, torch.complex128

        y_coord = torch.arange(npts_y, dtype=ftype) * dx_y
        x_coord = torch.arange(npts_x, dtype=ftype) * dx_x
        signal = make_signal_2d(x_coord, y_coord)

        fy_start, fy_end = fft_bin_aligned_range(npts_y, dx_y, 3, mpts_y)
        fx_start, fx_end = fft_bin_aligned_range(npts_x, dx_x, 5, mpts_x)

        czt = ChirpZTransform2D(
            num_input_points=(npts_y, npts_x),
            num_output_points=(mpts_y, mpts_x),
            input_step=(dx_y, dx_x),
            start_frequency=(fy_start, fx_start),
            end_frequency=(fy_end, fx_end),
            ftype=ftype,
            ctype=ctype,
        )
        with torch.no_grad():
            out = czt(signal)

        assert out.shape == (mpts_y, mpts_x)

        expected = torch.fft.fft2(signal)[3:3 + mpts_y, 5:5 + mpts_x]
        assert torch.allclose(out, expected, atol=1e-10, rtol=1e-10)

    def test_batched(self):
        """CZT2D should operate across a batch"""
        npts = 48
        mpts = 24
        dx = 0.02
        ftype, ctype = torch.float64, torch.complex128

        coord = torch.arange(npts, dtype=ftype) * dx
        s0 = make_signal_2d(coord, coord)
        s1 = make_signal_2d(coord + 0.05, coord - 0.03)
        batch = torch.stack([s0, s1]).unsqueeze(0)  # shape [1, 2, npts, npts]

        czt = ChirpZTransform2D(
            num_input_points=npts,
            num_output_points=mpts,
            input_step=dx,
            start_frequency=1.0,
            end_frequency=5.0,
            ftype=ftype,
            ctype=ctype,
        )
        with torch.no_grad():
            out_batch = czt(batch)
            out_0 = czt(s0)
            out_1 = czt(s1)

        assert out_batch.shape == (1, 2, mpts, mpts)
        assert torch.allclose(out_batch[0, 0], out_0)
        assert torch.allclose(out_batch[0, 1], out_1)
