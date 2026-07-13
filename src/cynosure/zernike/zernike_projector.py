import torch
import torch.nn as nn

from .zernike_polynomials import generate_zernike_polynomial, get_noll_index
from ..config import ZernikeConfig


class ZernikeProjector(nn.Module):
    def __init__(
            self,
            zernike_config: ZernikeConfig,
            r_norm: torch.Tensor,
            phi: torch.Tensor
    ):
        """
        `r_norm` and `phi` are both [H, W] coordinate tensors. `r_norm` is normalized\
        to 1 in the cardinal directions, sqrt(2) in the corners.
        """
        super().__init__()
        self.register_buffer("r_norm", r_norm)
        self.register_buffer("phi", phi)
        self._setup_zernike_basis(zernike_config)

    def _setup_zernike_basis(self, zernike_cfg: ZernikeConfig) -> None:
        """
        Builds a [num_elements, H, W] tensor of precalculated Zernike polynomials in the pupil plane.
        Also constructs an array of the corresponding Zernike indices. Both this and the Zernike
        bank are ordered by increasing Noll index.
        """

        max_n = zernike_cfg.max_n
        allowed_nm = set(zernike_cfg.allowed_nm)

        zernike_list = []
        for n in range(max_n+1):
            for m in range(-n, n+1, 2):
                j = get_noll_index(n, m)
                z = generate_zernike_polynomial(self.r_norm, self.phi, n, m, mask_to_unit_disk=True)
                if allowed_nm and not (n, m) in allowed_nm:
                    continue
                zernike_list.append((j, n, m, z))
        zernike_list = sorted(zernike_list, key=lambda x: x[0])  # sort by Noll index

        nm_indices = torch.tensor([item[1:3] for item in zernike_list], dtype=torch.long)
        zernike_bank = torch.stack([item[3] for item in zernike_list])
        self.register_buffer("zernike_bank", zernike_bank)
        self.register_buffer("nm_indices", nm_indices)
        assert zernike_bank.shape[0] == zernike_cfg.num_elements, (
            f"Zernike bank has {zernike_bank.shape[0]} elements; config says {zernike_cfg.num_elements}"
        )  # defensive
        self.num_elements = zernike_cfg.num_elements

    def project(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Projects the [B, N] tensor of Zernike coefficients onto `zernike_bank` and
        returns the constructed function as a [B, H, W] tensor.
        """
        B, N = coefficients.shape
        assert N == self.num_elements, f"ZernikeProjector expected {self.num_elements} coefficients but received {N}"
        out = torch.einsum("bn,nhw->bhw", coefficients.to(self.zernike_bank.dtype), self.zernike_bank)
        return out

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Alias for `project()`"""
        return self.project(coefficients)
