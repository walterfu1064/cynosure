"""
Module for reconstructing a z-stack by fitting its aberrations.

Encapsulates management of parameters for the phase aberrations and (optionally)
the amplitude aberrations, axial fit, and background fit. Also handles convolution
of the fitted PSF with a known object distribution.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Parameter

from . import BeamPropagator
from ..config import SimulationConfig, OpticalConfig, ZernikeConfig
from ..utilities.fft_utilities import convolve_psf_with_object


class ForwardZstack(nn.Module):
    def __init__(
            self,
            sim_cfg: SimulationConfig,
            optics_cfg: OpticalConfig,
            phase_cfg: ZernikeConfig,
            z_objective: torch.Tensor,
            object_distribution: torch.Tensor,
            *,
            amp_cfg: Optional[ZernikeConfig] = None,
            max_z_adjust: float = 0.5,
            fit_zscale: bool = True,
            fit_background: bool = True,
            initial_phase_speckle: float = 1.0e-6,
            initial_amplitude_speckle: float = 1.0e-6
    ):
        """
        Arguments:
        - sim_cfg: simulation configuration
        - optics_cfg: optical system configuration
        - phase_cfg: phase aberration configuration
        - z_objective: physical objective/stage positions, as reported by the microscope
        - object_distribution: tensor representing the emitter in the object space
        Kwargs:
        - amp_cfg: (optional) amplitude aberration configuration
        - max_z_adjust: max distance the fitted global defocus may go in either direction
        - fit_zscale: if True, include an ad hoc fitted z-scaling factor
        - fit_background: if True, include an ad hoc fitted, per-z, background
        - initial_phase_speckle, initial_amplitude_speckle: controls the strength of the initial phase/amp coef noise
        """

        super().__init__()
        self.propagator = BeamPropagator(sim_cfg, optics_cfg, phase_cfg, amp_cfg=amp_cfg)
        self.z_objective = z_objective
        self.num_z = z_objective.shape[-1]
        self.object_distribution = object_distribution
        self.max_z_adjust = max_z_adjust  # max fittable focus shift in either direction
        self.fit_zscale = fit_zscale
        self.fit_background = fit_background
        self.initial_phase_speckle = initial_phase_speckle
        self.initial_amplitude_speckle = initial_amplitude_speckle

        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initializes fittable parameters"""
        self.z_adjust_logit = Parameter(torch.zeros((1,), dtype=torch.float32))  # handles global defocus

        if self.fit_zscale:
            self.log_zscale = Parameter(torch.zeros((1,), dtype=torch.float32))  # residual axial scale, exp(log_scale) should land near 1
        else:
            self.log_zscale = None

        if self.fit_background:
            self.bg_logit = Parameter(torch.zeros((self.num_z, 1, 1), dtype=torch.float64))  # handles per-z background
        else:
            self.bg_logit = None

        phase_noise_factor = math.sqrt(self.initial_phase_speckle / self.propagator.num_phase_coefs)
        init_phase_coefs = torch.randn((1, self.propagator.num_phase_coefs)) * phase_noise_factor
        self.phase_coefs = Parameter(init_phase_coefs.clone())

        if self.num_amp_coefs > 0:
            amp_noise_factor = math.sqrt(self.initial_amplitude_speckle / self.propagator.num_amp_coefs)
            init_amp_coefs = torch.randn((1, self.propagator.num_amp_coefs)) * amp_noise_factor
            init_amp_coefs[:, 0] = 1.0  # bias towards a coherent spot
            self.amp_coefs = Parameter(init_amp_coefs.clone())
        else:
            self.amp_coefs = Parameter(torch.zeros((1, 0)))

    def reset(self):
        self._initialize_parameters()

    @property
    def num_phase_coefs(self) -> int:
        """Convenience pass-through"""
        return self.propagator.num_phase_coefs

    @property
    def num_amp_coefs(self) -> int:
        """Convenience pass-through"""
        return self.propagator.num_amp_coefs

    def get_recalibrated_z(self) -> torch.Tensor:
        """Scales and shifts the nominal z-positions by the learned parameters"""
        z_defocus = self.propagator.defocus_from_objective_z(self.z_objective)  # objective travel -> in-medium defocus
        zs = torch.exp(self.log_zscale) if self.fit_zscale else 1
        z0 = self.max_z_adjust * torch.tanh(self.z_adjust_logit)
        z = zs * z_defocus + z0
        return z

    def _apply_learned_image_normalization(self, image: torch.Tensor) -> torch.Tensor:
        """Normalizes a [..., H, W] image stack using a learned background factor"""
        image = image / image.sum((-2, -1), keepdim=True)
        if self.bg_logit is not None:
            num_pixels = image.size(-2) * image.size(-1)
            bg = torch.sigmoid(self.bg_logit)
            image = (1 - bg) * image + bg / num_pixels
        return image

    def forward(self) -> torch.Tensor:
        """
        Calculates the predicted z-stack using the stored aberrations and other fit parameters.

        Returns:
        - [B, H, W] tensor of calculated object intensities
        """
        z = self.get_recalibrated_z()
        psf = self.propagator(z, self.phase_coefs, self.amp_coefs)
        image = convolve_psf_with_object(self.object_distribution, psf)
        return self._apply_learned_image_normalization(image)
