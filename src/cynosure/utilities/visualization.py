import numpy as np
import matplotlib.pyplot as pyplot

import torch

from ..beam_propagation.beam_propagator import BeamPropagator


def plot_stack(stack: np.ndarray, panel_size: float = 3):
    """Plots a [N, H, W] image stack along a row of panels"""
    num_z = stack.shape[0]
    figsize = (panel_size * num_z, panel_size)
    fig, axarr = pyplot.subplots(1, num_z, figsize=figsize)
    for i, ax in enumerate(axarr):
        ax.imshow(stack[i], vmin=0)
    for ax in axarr.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()


def plot_field(
        propagator: BeamPropagator,
        amp_coefs: torch.Tensor,
        phase_coefs: torch.Tensor,
        panel_size: float = 3
):
    """Plots the amplitude and phase of the aperture field aberrated as given"""
    pupil_mask = propagator.pupil_mask.numpy(force=True)

    amplitude = propagator.get_aperture_amplitude(amp_coefs).abs().numpy(force=True).squeeze()
    amplitude = np.where(pupil_mask, amplitude, np.nan)

    phase = propagator.get_aperture_phase(phase_coefs).numpy(force=True).squeeze()
    phase = np.where(pupil_mask, phase, np.nan)

    fig, axarr = pyplot.subplots(1, 2, figsize=(2*panel_size, panel_size))
    ax1, ax2 = axarr
    ax1.imshow(amplitude)
    ax2.imshow(phase)
    for ax in axarr.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    ax1.set_title("amplitude")
    ax2.set_title("phase")
    fig.tight_layout()
