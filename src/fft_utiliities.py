import torch


def next_power_of_2(n: int) -> int:
    """Returns the smallest power of 2 that is >= `n`"""
    p = 1
    while p < n:
        p *= 2
    return p


def convolve_psf_with_object(object_distribution: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """
    Convolves a calculated PSF with a grayscale object distribution.

    Assumes `object_distribution` is a [H, W] floating point tensor, while `psf` is
    a [..., H, W] floating point tensor with its origin at (H//2, W//2) (i.e., the
    natural order for viewing, but not the natuarl FFT order).
    """
    H, W = object_distribution.shape[-2:]
    pH, pW = psf.shape[-2:]
    assert (pH, pW) == (H, W), f"Object spatial dimensions ({H}, {W}) do not match the PSF spatial grid ({pH}, {pW})"

    fft_size = (next_power_of_2(2 * H - 1), next_power_of_2(2 * W - 1))  # pad to avoid circular conv

    psf_padded = psf.new_zeros(*psf.shape[:-2], *fft_size)
    psf_padded[..., :H, :W] = psf
    psf_padded = torch.roll(psf_padded, shifts=(-(H // 2), -(W // 2)), dims=(-2, -1))
    psf_fft = torch.fft.rfft2(psf_padded, dim=(-2, -1))

    obj_fft = torch.fft.rfft2(object_distribution, s=fft_size, dim=(-2, -1))
    result = torch.fft.irfft2(obj_fft * psf_fft, s=fft_size, dim=(-2, -1))
    return result[..., :H, :W]

