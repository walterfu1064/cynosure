import math

import numpy as np


def get_radial_series(n: int, m: int) -> tuple[list, list]:
    """
    Returns a list of coefficients and corresponding powers for the radial
    part of the Zernike polynomial of the given order.
    """
    m = abs(m)
    assert n >= m, f"Invalid Zernike order: {n = }, {m = }"
    assert (n - m) % 2 == 0, f"Invalid Zernike order: {n = }, {m = }"

    def sign(k: int) -> float:
        return 1 if k % 2 == 0 else -1

    def numer(k: int) -> float:
        return math.factorial(n - k)

    def denom(k: int) -> float:
        return math.factorial(k) * math.factorial( ((n+m)//2) - k ) * math.factorial( ((n-m)//2) - k )

    k_iter = range((n - m)//2 + 1)
    coefs = [sign(k) * numer(k) / denom(k) for k in k_iter]
    powers = [n - 2*k for k in k_iter]
    return coefs, powers


def get_radial_term(r: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    Returns the radial part of the (n, m) Zernike polynomial as a 2D array.
    """
    coefs, powers = get_radial_series(n, m)
    coefs = np.asarray(coefs)
    powers = np.asarray(powers)
    return (coefs * np.power(r[..., None], powers)).sum(-1)


def get_angular_term(theta: np.ndarray, m: int) -> np.ndarray:
    """
    Returns the angular part of the (n, m) Zernike polynomial as a 2D array.
    """
    if m > 0:
        return np.cos(theta)
    elif m < 0:
        return np.sin(theta)
    else:
        return np.ones_like(theta)


def generate_zernike_polynomial(
        r: np.ndarray,
        theta: np.ndarray,
        n: int,
        m: int
) -> np.ndarray:
    """
    Returns the (n, m) Zernike polynomial evaluated over the 2D grids r and theta.
    Note that the result uses analytic continuation over the full domain of r and theta,
    and should be masked by the unit disk downstream.
    """
    rho = get_radial_term(r, n, m)
    gamma = get_angular_term(theta, m)
    return rho * gamma


def get_noll_index(n: int, m: int) -> int:
    """
    Returns the Noll index for Zernike polynomial (n, m).
    This is used to obtain a strict ordering over the polynomials.
    See:
        ```
        Noll, "Zernike polynomials and atmospheric turbulence," Journal of
        the Optical Society of America 66.3 pp. 207-211 (1976).
        ```
    """
    if m > 0 and n % 4 < 2:
        mod_term = 0
    elif m < 0 and n % 4 >= 2:
        mod_term = 0
    elif m >= 0 and n % 4 >= 2:
        mod_term = 1
    else:
        mod_term = 1
    return n*(n + 1)//2 + abs(m) + mod_term
