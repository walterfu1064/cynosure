"""
Forwards simulation module which simulates the image of an object through
an aberrated system by first calculating the PSF, then convolving it with
the object distribution.
"""

from typing import Optional

import torch
import torch.nn as nn

from beam_propagation import BeamPropagator
from fft_utiliities import convolve_psf_with_object


class ForwardModel(nn.Module):
    def __init__(self, propagator: BeamPropagator):
        super().__init__()
        self.propagator = propagator

    def forward(
            self,
            z: torch.Tensor,
            object_distribution: torch.Tensor,
            phase_coefs: torch.Tensor,
            amp_coefs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Given a batch of propagation distances and object distributions, along with
        phase Zernike coefficients, and (optionally) amplitude Zernike coefficients,
        simulates how the objects will appear through the described imaging systems.

        Arguments:
        - z: [B,] tensor of propagation distances, measured from the focal plane
        - object_distribution: [B, object_grid, object_grid] floating point object distribution
        - phase_coefs: [B, N_phase] tensor of Zernike coefficients used to phase-aberrate the aperture field
        - amp_coefs: [B, N_phase] tensor of Zernike coefficients used to amplitude-aberrate the aperture field
            (Required if and only if `aperture_type` is `fitted`)

        Returns:
        - [B, H, W] tensor of calculated object intensities
        """
        psf = self.propagator(z, phase_coefs, amp_coefs)
        return convolve_psf_with_object(object_distribution, psf)
