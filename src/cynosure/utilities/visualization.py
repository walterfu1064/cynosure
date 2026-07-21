from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as pyplot
import torch

from ..beam_propagation.beam_propagator import BeamPropagator


def format_zernike_names(nm_indices: Union[np.ndarray, torch.Tensor], symbol: str = "Z") -> list[str]:
    """Takes a [N, 2] array/tensor of Zernike indices and formats them as `Z_n^m` (or some other symbol)"""
    if isinstance(nm_indices, torch.Tensor):
        nm_indices = nm_indices.numpy(force=True)
    if not isinstance(nm_indices, np.ndarray):
        nm_indices = np.asarray(nm_indices)
    nm_list = nm_indices.astype(np.int64).tolist()
    return [r"$%s_{%d}^{%d}$" % (symbol, n, m) for n, m in nm_list]


def plot_stack(stack: np.ndarray, panel_size: float = 3):
    """Plots a [N, H, W] image stack along a row of panels"""
    num_z = stack.shape[0]
    figsize = (panel_size * num_z, panel_size)
    fig, axarr = pyplot.subplots(1, num_z, figsize=figsize, squeeze=False)
    for i, ax in enumerate(axarr.ravel()):
        ax.imshow(stack[i], vmin=0)
    for ax in axarr.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


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
    return fig


def plot_correlation_matrix(
        correlation: np.ndarray,
        labels: list[str],
        num_phase: Optional[int] = None,
        panel_size: float = 7,
):
    """
    Plots a correlation matrix all pretty-like.

    Arguments:
    - correlations: [N, N] matrix with values in [-1, 1]
    - labels: [N,] list of variable names
    - num_phase: if provided, add guidelines to divide phase vs. amplitude parts
    - panel_size: size of panel/figure
    """
    correlation = correlation.astype(np.float32).copy()  # copy because we're going to modify
    np.fill_diagonal(correlation, np.nan)

    cmap = pyplot.get_cmap("RdBu_r").copy()  # copy because we're going to modify
    cmap.set_bad("0.9")  # mask diagonal

    fig, ax = pyplot.subplots(figsize=(panel_size * 1.1, panel_size))
    im = ax.imshow(correlation, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    if num_phase is not None:
        ax.axhline(num_phase - 0.5, color="k", linewidth=0.8, alpha=0.5)
        ax.axvline(num_phase - 0.5, color="k", linewidth=0.8, alpha=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    fig.tight_layout()
    return fig


def plot_coefficient_uncertainties(
        sigmas: np.ndarray,
        prior_sigmas: np.ndarray,
        labels: list[str],
        panel_size: float = 3,
):
    """
    Plots a bar chart of the predicted per-coefficient uncertainties, overlaid with the
    prior uncertainties as a reference for how strongly the images constrained each one.
    """
    x = np.arange(len(labels))
    fig, ax = pyplot.subplots(figsize=(max(2 * panel_size, 0.3 * len(labels)), panel_size))
    ax.bar(x, sigmas, color="c", label="predicted")
    ax.plot(x, prior_sigmas, linestyle="none", marker="_", markersize=11, color="r", label="prior")
    ax.set_xticks(x, labels, rotation=90)
    ax.set_xlim(-0.75, len(labels)-0.25)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("coefficient sigma")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
