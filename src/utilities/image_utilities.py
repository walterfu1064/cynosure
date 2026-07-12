import numpy as np
from scipy import ndimage


def laplacian_variance(image: np.ndarray):
    """Returns the Laplacian variance of an image along its last two axes (assumed spatial)"""
    kernel = np.array([1.0, -2.0, 1.0])
    image = image.astype(np.float64)
    laplacian = ndimage.correlate1d(image, kernel, axis=-2) + ndimage.correlate1d(image, kernel, axis=-1)
    return laplacian.var((-2, -1))


def blur_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian-blurring for a 1d array.
    Intended for use in smoothing out the Laplacian variance of
    a z-stack so the focal plane can be estimated a bit more robustly.
    Ultimately a convenience, not load-bearing.
    """
    return ndimage.gaussian_filter(arr, sigma=sigma, mode="nearest")
