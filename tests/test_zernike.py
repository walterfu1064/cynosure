"""
Unit tests for zernike.py.
"""

import math

import pytest
import torch

from zernike import generate_zernike_polynomial, get_noll_index


# Canonical Noll (j, n, m) triples:
NOLL_TABLE = [
    (1,  0,  0),
    (2,  1,  1),
    (3,  1, -1),
    (4,  2,  0),
    (5,  2, -2),
    (6,  2,  2),
    (7,  3, -1),
    (8,  3,  1),
    (9,  3, -3),
    (10, 3,  3),
    (11, 4,  0),
    (12, 4,  2),
    (13, 4, -2),
    (14, 4,  4),
    (15, 4, -4),
]


class TestNollIndex:
    @pytest.mark.parametrize("j, n, m", NOLL_TABLE)
    def test_canonical_values(self, j, n, m):
        assert get_noll_index(n, m) == j

    @pytest.mark.parametrize("n_max", [4, 6, 8, 10])
    def test_bijection(self, n_max: int):
        pairs = [(n, m) for n in range(n_max + 1) for m in range(-n, n + 1, 2)]
        js = [get_noll_index(n, m) for (n, m) in pairs]
        assert sorted(js) == list(range(1, len(pairs) + 1))


class TestKnownPolynomials:
    @pytest.fixture
    def grid(self):
        r = torch.linspace(0, 1, 25, dtype=torch.float64)
        phi = torch.linspace(0, 2 * math.pi, 25, dtype=torch.float64)
        R, PHI = torch.meshgrid(r, phi, indexing="ij")
        return R, PHI

    def test_piston(self, grid):
        R, PHI = grid
        z = generate_zernike_polynomial(R, PHI, 0, 0, mask_to_unit_disk=False)
        assert torch.allclose(z, torch.ones_like(R), atol=1e-12)

    def test_tip(self, grid):
        # Z(1, 1) = 2 r cos(phi)
        R, PHI = grid
        z = generate_zernike_polynomial(R, PHI, 1, 1, mask_to_unit_disk=False)
        expected = 2 * R * torch.cos(PHI)
        assert torch.allclose(z, expected, atol=1e-12)

    def test_tilt(self, grid):
        # Z(1, -1) = 2 r sin(phi)
        R, PHI = grid
        z = generate_zernike_polynomial(R, PHI, 1, -1, mask_to_unit_disk=False)
        expected = 2 * R * torch.sin(PHI)
        assert torch.allclose(z, expected, atol=1e-12)

    def test_defocus(self, grid):
        # Z(2, 0) = sqrt(3) (2 r^2 - 1)
        R, PHI = grid
        z = generate_zernike_polynomial(R, PHI, 2, 0, mask_to_unit_disk=False)
        expected = math.sqrt(3) * (2 * R ** 2 - 1)
        assert torch.allclose(z, expected, atol=1e-12)

    def test_astigmatism_vertical(self, grid):
        # Z(2, 2) = sqrt(6) r^2 cos(2 phi)
        R, PHI = grid
        z = generate_zernike_polynomial(R, PHI, 2, 2, mask_to_unit_disk=False)
        expected = math.sqrt(6) * R ** 2 * torch.cos(2 * PHI)
        assert torch.allclose(z, expected, atol=1e-12)

    def test_astigmatism_oblique(self, grid):
        # Z(2, -2) = sqrt(6) r^2 sin(2 phi)
        R, PHI = grid
        z = generate_zernike_polynomial(R, PHI, 2, -2, mask_to_unit_disk=False)
        expected = math.sqrt(6) * R ** 2 * torch.sin(2 * PHI)
        assert torch.allclose(z, expected, atol=1e-12)

    def test_coma_horizontal(self, grid):
        # Z(3, 1) = sqrt(8) (3 r^3 - 2 r) cos(phi)
        R, PHI = grid
        z = generate_zernike_polynomial(R, PHI, 3, 1, mask_to_unit_disk=False)
        expected = math.sqrt(8) * (3 * R ** 3 - 2 * R) * torch.cos(PHI)
        assert torch.allclose(z, expected, atol=1e-12)

    def test_trefoil_oblique(self, grid):
        # Z(3, 3) = sqrt(8) r^3 cos(3 phi)
        R, PHI = grid
        z = generate_zernike_polynomial(R, PHI, 3, 3, mask_to_unit_disk=False)
        expected = math.sqrt(8) * R ** 3 * torch.cos(3 * PHI)
        assert torch.allclose(z, expected, atol=1e-12)

    def test_primary_spherical(self, grid):
        # Z(4, 0) = sqrt(5) (6 r^4 - 6 r^2 + 1)
        R, PHI = grid
        z = generate_zernike_polynomial(R, PHI, 4, 0, mask_to_unit_disk=False)
        expected = math.sqrt(5) * (6 * R ** 4 - 6 * R ** 2 + 1)
        assert torch.allclose(z, expected, atol=1e-12)


class TestOrthonormality:
    @staticmethod
    def polar_grid(n_r: int = 400, n_phi: int = 512):
        r_edges = torch.linspace(0, 1, n_r + 1, dtype=torch.float64)
        r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
        dr = 1.0 / n_r
        phi_edges = torch.linspace(0, 2 * math.pi, n_phi + 1, dtype=torch.float64)
        phi_mid = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        dphi = 2 * math.pi / n_phi
        R, PHI = torch.meshgrid(r_mid, phi_mid, indexing="ij")
        weights = R * dr * dphi
        return R, PHI, weights

    def test_unit_norm(self):
        R, PHI, W = self.polar_grid()
        for n in range(5):
            for m in range(-n, n + 1, 2):
                z = generate_zernike_polynomial(R, PHI, n, m, mask_to_unit_disk=False)
                norm_sq = (z * z * W).sum().item() / math.pi
                assert abs(norm_sq - 1.0) < 1e-4, f"Z(n={n}, m={m}) norm^2 = {norm_sq}"
