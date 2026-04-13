"""
PyTorch module for simulating beam propagation.

Intended usage is to, given a set of aberrations at the pupil plane (e.g., immediately after
an objective lens), propagate the field by some distance to near the object plane.

Simulations uses results from Debye-Wolf theory:
    ```
    Wolf, "Electromagnetic diffraction in optical systems - I. An integral representation of the
    image field," Proceedings of the Royal Society of London, Series A, Mathematical and Physical
    Sciences, 253.1274, pp. 349-357 (1959).
    royalsocietypublishing.org/rspa/article-abstract/253/1274/349/10631
    ```
and Richards-Wolf theory:
    ```
    Richards and Wolf, "Electromagnetic diffraction in optical systems - II. Structure of the image
    field in an aplanatic system," Proceedings of the Royal Society of London, Series A,
    Mathematical and Physical Sciences, 253.1274, pp. 349-357 (1959).
    royalsocietypublishing.org/rspa/article-abstract/253/1274/358/10622
    ```
and draws heavy inspiration from:
    ```
    Vishniakou and Seelig, "Differentiable optimization of the Debye-Wolf integral for light
    shaping and adaptive optics in two-photon microscopy," Optics Express 31.6 pp. 9526-9542 (2023).
    https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-6-9526
    ```

The use of Debye-Wolf includes the following approximations:
- light is monochromatic
- evanescent waves are negligible
- the pupil and the propagation distance are both much larger than the wavelength (so Kirchhoff boundaries apply near the pupil)
"""


from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from czt import ChirpZTransform2D
from zernike import generate_zernike_polynomial, get_noll_index


@dataclass
class SimulationConfig:
    pupil_grid_size: int
    object_grid_size: int
    object_pixel_size: float


@dataclass
class OpticalConfig:
    wavelength: float
    focal_length: float
    numerical_aperture: float


class GridCollection(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    r: torch.Tensor
    phi: torch.Tensor


class BeamPropagator(nn.Module):
    def __init__(
            self,
            sim_cfg: SimulationConfig,
            optics_cfg: OpticalConfig,
            max_zernike_n: int,
            *,
            ftype: torch.dtype = torch.float64,
            ctype: torch.dtype = torch.complex128,
    ):
        super().__init__()

        self.pupil_grid_size = sim_cfg.pupil_grid_size
        self.object_grid_size = sim_cfg.object_grid_size
        self.object_dx = sim_cfg.object_pixel_size

        self.wavelength = optics_cfg.wavelength
        self.k0 = 2 * torch.pi / self.wavelength
        self.focal_length = optics_cfg.focal_length
        self.numerical_aperture = optics_cfg.numerical_aperture

        self.ftype = ftype
        self.ctype = ctype

        self._setup_pupil_coordinates()
        self._setup_object_coordinates()
        self._setup_polarization_factors()
        self._setup_zernike_basis(max_zernike_n)

        start_frequency, stop_frequency = self._calculate_object_frequencies()
        self.czt = ChirpZTransform2D(
            num_input_points=self.pupil_grid_size,
            num_output_points=self.object_grid_size,
            input_step=self.pupil_dx,
            start_frequency=start_frequency,
            end_frequency=stop_frequency,
            ftype=ftype,
            ctype=ctype,
        )

    @staticmethod
    def _construct_normalized_coordinates(grid_size: int, dtype: torch.dtype) -> GridCollection:
        """Returns rectilinear and polar coordinate grids covering the unit square/disk"""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, grid_size, dtype=dtype),
            torch.linspace(-1, 1, grid_size, dtype=dtype),
            indexing="ij",
        )
        r = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)
        return GridCollection(x, y, r, phi)

    def _setup_pupil_coordinates(self) -> None:
        """
        Saves rectilinear and polar coordinate grids as buffers, both in normalized
        units as well as spanning the pupil.
        """
        pupil_grids = self._construct_normalized_coordinates(self.pupil_grid_size, self.ftype)

        self.register_buffer("pupil_x_norm", pupil_grids.x)
        self.register_buffer("pupil_y_norm", pupil_grids.y)
        self.register_buffer("pupil_r_norm", pupil_grids.r)
        self.register_buffer("pupil_phi", pupil_grids.phi)
        self.register_buffer("pupil_mask", self.pupil_r_norm <= 1)

        self.pupil_radius = self.focal_length * self.numerical_aperture
        self.pupil_dx = 2 * self.pupil_radius / (self.pupil_grid_size - 1)
        self.register_buffer("pupil_x", self.pupil_x_norm * self.pupil_radius)
        self.register_buffer("pupil_y", self.pupil_y_norm * self.pupil_radius)
        self.register_buffer("pupil_r", self.pupil_r_norm * self.pupil_radius)

    def _setup_object_coordinates(self) -> None:
        """
        Saves rectilinear and polar coordinate grids as buffers, both in normalized
        units as well as spanning the object space.
        """
        object_grids = self._construct_normalized_coordinates(self.object_grid_size, self.ftype)

        self.register_buffer("object_x_norm", object_grids.x)
        self.register_buffer("object_y_norm", object_grids.y)
        self.register_buffer("object_r_norm", object_grids.r)
        self.register_buffer("object_phi", object_grids.phi)

        self.object_radius = (self.object_grid_size - 1) * self.object_dx / 2
        self.register_buffer("object_x", self.object_x_norm * self.object_radius)
        self.register_buffer("object_y", self.object_y_norm * self.object_radius)
        self.register_buffer("object_r", self.object_r_norm * self.object_radius)

    def _setup_polarization_factors(self) -> None:
        """
        Constructs a polarization rotation matrix per Richards-Wolf theory.
        Matrix has shape [3, 2, pupil_grid, pupil_grid], corresponding to the
        (x, y, z) rotated components of the (x, y) initial polarizations.

        Requires `_setup_pupil_coordinates` to have already been called.
        """

        # Spherical coordinates in the pupil frame (z along the optical axis, phi azimuthal, theta polar):
        sin_theta = torch.clamp(self.numerical_aperture * self.pupil_r_norm, max=1.0)
        cos_theta = torch.sqrt(torch.clamp(1 - sin_theta**2, min=0.0))
        cos_phi = torch.cos(self.pupil_phi)
        sin_phi = torch.sin(self.pupil_phi)

        c_minus_1 = cos_theta - 1.0
        shape = (3, 2) + cos_theta.shape
        M = torch.zeros(shape, dtype=self.pupil_r_norm.dtype, device=self.pupil_r_norm.device)
        M[0, 0] = 1.0 + c_minus_1 * cos_phi ** 2
        M[0, 1] = c_minus_1 * sin_phi * cos_phi
        M[1, 0] = c_minus_1 * sin_phi * cos_phi
        M[1, 1] = 1.0 + c_minus_1 * sin_phi ** 2
        M[2, 0] = -sin_theta * cos_phi
        M[2, 1] = -sin_theta * sin_phi

        M = M * torch.sqrt(cos_theta)  # aplanatic apodization for energy conservation
        M = M * self.pupil_mask.to(M.dtype)  # reassert pupil mask
        self.register_buffer("polarization_rot", M)

    def _setup_zernike_basis(self, max_zernike_n: int) -> None:
        """
        Builds a [num_zernikes, H, W] tensor of precalculated Zernike polynomails in the pupil plane.
        Also constructs an array of the corresponding Zernike indices. Both this and the Zernike
        bank are ordered by increasing Noll index.
        """
        zernike_list = []
        for n in range(max_zernike_n+1):
            for m in range(-n, n+1, 2):
                j = get_noll_index(n, m)
                z = generate_zernike_polynomial(self.pupil_r_norm, self.pupil_phi, n, m, mask_to_unit_disk=True)
                zernike_list.append((j, n, m, z))
        zernike_list = sorted(zernike_list, key=lambda x: x[0])  # sort by Noll index

        nm_indices = torch.tensor([item[1:3] for item in zernike_list], dtype=torch.long)
        zernike_bank = torch.stack([item[3] for item in zernike_list])
        self.register_buffer("zernike_bank", zernike_bank)
        self.register_buffer("nm_indices", nm_indices)

    def _calculate_object_frequencies(self) -> tuple[float, float]:
        """
        Returns the CZT start/end frequencies that sample the object grid.

        Follows from the fact that Debye-Wolf reduces to a 2D Fourier transform bewteen
        the pupil coordinate x_p and the transverse spatial frequency f_xy = x_o / (wl * focal).
        """
        max_frequency = self.object_radius / (self.wavelength * self.focal_length)
        return -max_frequency, max_frequency

    def _make_aperture_field(self, aberrations: torch.Tensor):
        """Construcst a complex field that fills the aperture and has the given aberration coefficients."""
        pass

    def _propagate(self, aperture_field: torch.Tensor, z: torch.Tensor):
        """Propagates the given field to the given z position(s)"""
        pass

    def forward(self, aberrations: torch.Tensor):
        pass

