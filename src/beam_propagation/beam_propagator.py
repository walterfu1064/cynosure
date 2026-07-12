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

The use of Debye-Wolf implies the following approximations:
- light is monochromatic
- evanescent waves are negligible
- the pupil and the propagation distance are both much larger than the wavelength (so Kirchhoff boundaries apply near the pupil)

Models a system where the objective is immersed in some medium `immersion_index`, and the
emitters are embedded in some medium `medium_index`. Input z-positions should represent the
defocus in the immersion medium, **not** the physical positions of the objective.
The `defocus_from_objective_z` function is provided to perform this conversion.
"""


from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from config import SimulationConfig, OpticalConfig, ZernikeConfig
from czt import ChirpZTransform2D
from zernike import ZernikeProjector


class GridCollection(NamedTuple):
    """Convenience to bundle rectilinear and polar coordinate grids"""
    x: torch.Tensor
    y: torch.Tensor
    r: torch.Tensor
    phi: torch.Tensor


class BeamPropagator(nn.Module):
    def __init__(
            self,
            sim_cfg: SimulationConfig,
            optics_cfg: OpticalConfig,
            phase_cfg: ZernikeConfig,
            *,
            amp_cfg: Optional[ZernikeConfig] = None,
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
        self.medium_index = optics_cfg.medium_index
        self.immersion_index = optics_cfg.immersion_index if optics_cfg.immersion_index is not None else self.medium_index
        self.aperture_type = optics_cfg.aperture_type

        if self.numerical_aperture > self.medium_index:
            raise ValueError(
                f"Numerical aperture ({self.numerical_aperture}) cannot exceed "
                f"the medium index ({self.medium_index})"
            )
        if self.numerical_aperture > self.immersion_index:
            raise ValueError(
                f"Numerical aperture ({self.numerical_aperture}) cannot exceed "
                f"the immersion index ({self.immersion_index})"
            )

        # Objective is immersed in n1, sample medium is n2 -> physically moving the objective
        # by dz moves the focal plane by (n1/n2)*dz, so we need to rescale z.
        # Identity scaling for indexed-matched systems (default when immersion_index == None).
        self.axial_scale = self.immersion_index / self.medium_index

        if self.aperture_type not in ("flat", "gaussian", "supergaussian", "fitted"):
            raise ValueError(
                f"Unknown aperture type: {self.aperture_type} "
                "(allowed options are `flat`, `gaussian`, `supergaussian`, `fitted`)"
            )

        self.ftype = ftype
        self.ctype = ctype

        self._setup_pupil_coordinates()
        self._setup_object_coordinates()
        self._setup_polarization_factors()
        self._setup_axial_wavenumbers()

        self.phase_projector = ZernikeProjector(phase_cfg, self.pupil_r_norm, self.pupil_phi)
        self.num_phase_coefs = self.phase_projector.num_elements

        if amp_cfg is not None:
            self.amp_projector = ZernikeProjector(amp_cfg, self.pupil_r_norm, self.pupil_phi)
            self.num_amp_coefs = self.amp_projector.num_elements
        else:
            self.amp_projector = None
            self.num_amp_coefs = 0

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

        # Spherical coordinates in the pupil frame (z along the optical axis, phi azimuthal, theta polar in medium):
        sin_theta = torch.clamp(self.numerical_aperture * self.pupil_r_norm / self.medium_index, max=1.0)
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

        safe_cos = torch.where(self.pupil_mask, cos_theta, torch.ones_like(cos_theta))
        M = M / torch.sqrt(safe_cos)  # aplanatic apodization combined with Jacobian
        M = M * self.pupil_mask.to(M.dtype)  # reassert pupil mask
        self.register_buffer("cos_theta", cos_theta)
        self.register_buffer("polarization_rot", M)

    def _setup_axial_wavenumbers(self) -> None:
        """
        Precomputes the axial wavenumber k_z(rho) = k0 * sqrt(n^2 - NA^2 * rho^2) that governs
        defocus in the focusing medium (n = `medium_index`). The transverse wavenumber
        NA * rho * k0 is conserved across  interfaces, so emissions from a bead in medium n
        evolve at this modified k_z.

        Requires `_setup_pupil_coordinates` to have already been called.
        """
        transverse = self.numerical_aperture * self.pupil_r_norm
        kz = self.k0 * torch.sqrt(torch.clamp(self.medium_index**2 - transverse**2, min=0.0))
        self.register_buffer("axial_wavenumber", kz)

    def _calculate_object_frequencies(self) -> tuple[float, float]:
        """
        Returns the CZT start/end frequencies that sample the object grid.

        Follows from the fact that Debye-Wolf reduces to a 2D Fourier transform bewteen
        the pupil coordinate x_p and the transverse spatial frequency f_xy = x_o / (wl * focal).
        """
        max_frequency = self.object_radius / (self.wavelength * self.focal_length)
        return -max_frequency, max_frequency

    def get_aperture_amplitude(self, amp_coefs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Constructs the field amplitude at the aperture plane as a real tensor.
        Field might be flat, gaussian, supergaussian (m=8), or fitted (using the given Zernike coefficients).
        Output is always masked by `self.pupil_mask` before returning.

        Flat aperture:
            Returns a [1, pupil_grid, pupil_grid] tensor of ones.
        Gaussian aperture:
            Returns a [1, pupil_grid, pupil_grid] tensor.
            Scaled to 50% intensity at r=0.5, 6.25% intensity at r=1.0.
        Supergaussian aperture:
            Returns a [1, pupil_grid, pupil_grid] tensor.
            Scaled to 50% intensity at r=0.75, 0.1% intensity at r=1.0.
        Zernike aperture:
            Requires `amp_coefs` to be passed as a [B, N] tensor.
            Returns a [B, pupil_grid, pupil_grid] tensor.
        """
        device = self.pupil_r_norm.device
        match self.aperture_type:
            case "flat":
                field = torch.ones((self.pupil_grid_size, self.pupil_grid_size), device=device)
            case "gaussian":
                field = torch.exp(-0.5 * torch.pow(self.pupil_r_norm * 1.665, 2)).unsqueeze(0)
            case "supergaussian":
                field = torch.exp(-0.5 * torch.pow(self.pupil_r_norm * 1.274, 8)).unsqueeze(0)
            case "fitted":
                if amp_coefs is None:
                    raise ValueError("Unable to construct fitted aperture field with amp_aberrations = None")
                field = self.amp_projector(amp_coefs)
            case _:
                raise ValueError(f"Unknown aperture type: {self.aperture_type}")
        return field * self.pupil_mask.to(self.ftype)

    def get_aperture_phase(self, phase_coefs: torch.Tensor) -> torch.Tensor:
        """
        Constructs the field phase at the aperture plane as a real tensor.
        Output is masked by `self.pupil_mask` before returning.
        """
        phase = self.phase_projector(phase_coefs)
        return phase * self.pupil_mask.to(self.ftype)

    def get_aperture_field(
            self,
            phase_coefs: torch.Tensor,
            amp_coefs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Combines the aperture amplitude and phase into a complex field"""
        amplitude = self.get_aperture_amplitude(amp_coefs)
        phase = self.get_aperture_phase(phase_coefs)
        field = amplitude * torch.exp(2j*torch.pi*phase)
        return field

    def defocus_from_objective_z(self, objective_z: torch.Tensor) -> torch.Tensor:
        """
        Converts nominal z-positions (physical objective travel) in the immersion medium to
        the in-medium defocus that `forward` and `_propagate` expect using the index-mismatch
        axial scale factor. No-op for index-matched systems.
        """
        return objective_z * self.axial_scale

    def _propagate(self, field: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Propagates the given (scalar) field to the given z positions.
        Takes `field` as [B, pupil_grid, pupil_grid], and `z` as [B,].
        `z` is the defocus in the focusing medium (see `defocus_from_objective_z`), measured from the focal plane.
        Returns a [B, output_pol, input_pol, object_grid, object_grid] batch of fields,
        separated by the 3 output polarizations for each of the 2 input polarizations.
        """
        B = field.shape[0]
        prop_phase = z.view(B, 1, 1) * self.axial_wavenumber  # [B, H, W]
        prop_field = field * torch.exp(1j * prop_phase)
        prop_field = prop_field.view(B, 1, 1, self.pupil_grid_size, self.pupil_grid_size)
        prop_field = prop_field * self.polarization_rot.unsqueeze(0)  # [B, 3, 2, H, W]
        prop_field = self.czt(prop_field.flatten(0, 2)).view(B, 3, 2, self.object_grid_size, self.object_grid_size)
        return prop_field

    def forward(
            self,
            z: torch.Tensor,
            phase_coefs: torch.Tensor,
            amp_coefs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Given a batch of propagation distances, phase Zernike coefficients, and (optionally)
        amplitude Zernike coefficients, constructs an aberrated aperture field and propagates it forwards.

        Arguments:
        - z: [B,] defocus in the focusing medium (see `defocus_from_objective_z`), measured from the focal plane
        - phase_coefs: [B, N_phase] tensor of Zernike coefficients used to phase-aberrate the aperture field
        - amp_coefs: [B, N_phase] tensor of Zernike coefficients used to amplitude-aberrate the aperture field
            (Required if and only if `aperture_type` is `fitted`)
        Returns:
        - [B, H, W] tensor of the total intensities at the propagation distances
        """
        aperture_field = self.get_aperture_field(phase_coefs, amp_coefs)  # [B, pupil_grid, pupil_grid]
        prop_field = self._propagate(aperture_field, z)  # [B, output_pol, input_pol, pupil_grid, pupil_grid]
        intensity = torch.abs(prop_field)**2
        return intensity.sum((1, 2)) / 2  # incoherent polarization sum, returns [B, object_grid, object_grid]
