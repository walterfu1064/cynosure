from dataclasses import dataclass, field

import numpy as np
import torch

from czt import ChirpZTransform2D
from zernike import generate_zernike_polynomial, get_noll_index


@dataclass
class PlaneConfig:
    """Coordinate configuration at a particular optical plane"""
    grid_size: int
    pixel_size: float


@dataclass
class OpticalConfig:
    """Optical configuration"""
    wavelength: float
    focal_length: float


class BeamPropagator:
    def __init__(
            self,
            pupil_cfg: PlaneConfig,
            object_cfg: PlaneConfig,
            max_zernike_n: int,
            *,
            ftype: torch.dtype = torch.float64,
            ctype: torch.dtype = torch.complex128,
    ):

        self.pupil_x, self.pupil_y, self.pupil_r, self.pupil_theta = self._setup_coordinates(pupil_cfg)
        self.pupil_mask = self.pupil_r <= 1
        self.object_x, self.object_y, self.object_r, self.object_theta = self._setup_coordinates(object_cfg)

        self.zernike_bank, self.nm_indices = self._build_zernike_basis(max_zernike_n)

        # FIXME - need to figure out how the object grid maps to start_ and end_frequency
        # self.czt = ChirpZTransform2D(
        #     num_input_points=(pupil_cfg.grid_size, pupil_cfg.grid_size),
        #     num_output_points=(object_cfg.grid_size, object_cfg.grid_size),
        #     input_step=(object_cfg.pixel_size, object_cfg.pixel_size),
        #     start_frequency=(None, None),
        #     end_frequency=(None, None),
        #     ftype=ftype,
        #     ctype=ctype,
        # )

    @staticmethod
    def _setup_coordinates(cfg: PlaneConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y, x = np.meshgrid(
            np.linspace(-1, 1, cfg.grid_size),
            np.linspace(-1, 1, cfg.grid_size)
        )
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        r = torch.from_numpy(r)
        theta = torch.from_numpy(theta)
        return x, y, r, theta

    def _build_zernike_basis(self, max_zernike_n: int) -> tuple[torch.Tensor, np.ndarray]:
        """
        Builds a [num_zernikes, H, W] tensor of precalculated Zernike polynomails in the pupil plane.
        Also returns an array of the corresponding Zernike indices. Both this and the Zernike bank
        are ordered by increasing Noll index.
        """
        zernike_list = []
        for n in range(max_zernike_n+1):
            for m in range(-n, n+1, 2):
                j = get_noll_index(n, m)
                zernike_list.append((j, n, m, generate_zernike_polynomial(self.pupil_r, self.pupil_theta, n, m)))
        zernike_list = sorted(zernike_list)  # sort by Noll index

        nm_indices = np.array([item[1:3] for item in zernike_list])
        zernike_bank = torch.stack([item[3] for item in zernike_list])
        return zernike_bank, nm_indices

    def _make_aperture_field(self):
        pass


