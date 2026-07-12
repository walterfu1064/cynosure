import math

import torch


def get_radial_series(n: int, m: int) -> tuple[list, list]:
    """
    Returns a list of coefficients and corresponding powers for the radial
    part of the Zernike polynomial of the given order.

    Note that past n ~ 12 or so, this method will lose precision, and should
    be replaced with a recurrence relation.
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


def get_radial_term(r: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """
    Returns the radial part of the (n, m) Zernike polynomial as a 2D array.
    """
    coefs, powers = get_radial_series(n, m)
    coefs = torch.as_tensor(coefs).to(r.dtype)
    powers = torch.as_tensor(powers).to(r.dtype)
    return (coefs * torch.pow(r[..., None], powers)).sum(-1)


def get_angular_term(phi: torch.Tensor, m: int) -> torch.Tensor:
    """
    Returns the angular part of the (n, m) Zernike polynomial as a 2D array.
    """
    if m > 0:
        return torch.cos(m*phi)
    elif m < 0:
        return torch.sin(abs(m)*phi)
    else:
        return torch.ones_like(phi)


def generate_zernike_polynomial(
        r: torch.Tensor,
        phi: torch.Tensor,
        n: int,
        m: int,
        mask_to_unit_disk: bool
) -> torch.Tensor:
    """
    Returns the (n, m) Zernike polynomial evaluated over the 2D grids r and phi.
    Polynomials are normalized to be orthonormal.

    If `mask_to_unit_disk` is True, results will be zeroed for all r > 1.
    If False, polynomials will be analytically continued over the full domain for and phi.
    """
    rho = get_radial_term(r, n, m)
    gamma = get_angular_term(phi, m)

    norm = math.sqrt(n+1) if m == 0 else math.sqrt(2*n+2)
    z = norm * rho * gamma

    if mask_to_unit_disk:
        z *= (r <= 1)
    return z


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
    base = n * (n + 1) // 2 + abs(m)
    if m == 0:
        return base + 1
    elif m > 0:
        if n % 4 < 2:
            return base
        else:
            return base + 1
    else:
        if n % 4 < 2:
            return base + 1
        else:
            return base
