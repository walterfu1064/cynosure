"""
Functions for assessing posterior head calibration. Follows the Simulation-Based
Calibration approach laid out by Talts et al., "Validating Bayesian inference
algorithms with simulation-based calibration," arXiv:1804.06788 (2018).

In brief, if the target and the posterior samples are from the same distribution,
then when they are ordered, the target should be equally likely to be in any rank.
Deviations from uniformity in the rank distribution are therefore signs that the
posterior distribution is not well-calibrated.
"""

import math
from typing import Optional

import numpy as np
from scipy import stats
import torch

from cynosure.zstack_decoding.zstack_solver import ZstackSolver


def get_expected_rank_statistics(
        batch_size: int,
        num_bins: int,
) -> tuple[float, float]:
    """
    Returns the expected mean and standard deviation if the rank statistics for
    `batch_size` draws, normalized to [0, 1] and split into `num_bins` bins,
    is uniformly distributed.
    """
    mean = batch_size / num_bins
    sdev = mean * math.sqrt((num_bins - 1) / batch_size)
    return mean, sdev


def calculate_rank_statistics(
        targets: torch.Tensor,
        samples: torch.Tensor,
) -> torch.Tensor:
    """
    Given a set of target coefficients (`batch_size` independent draws) and
    a set of sampled posterior predictions (`num_samples` per draw), calculates
    the rank statistics for each variable.

    Rank statistics are normalized to [0, 1]. A target variable gets 0 if it's
    smaller than all the samples, or 1 if it's larger than all of them. The
    distribution of rank statistics should be uniform for a well-calibrated posterior.

    Arguments:
    - targets: [batch_size, num_variables]
    - samples: [batch_size, num_samples, num-variables]
    Returns:
    - ranks: [batch_size, num_variables] rankings in [0, 1]
    """
    if targets.shape[0] != samples.shape[0]:
        raise ValueError(
            f"Target and samples have different batch sizes {targets.shape[0]} and {samples.shape[0]}"
        )
    if targets.shape[-1] != samples.shape[-1]:
        raise ValueError(
            f"Target and samples have different variable counts {targets.shape[-1]} and {samples.shape[-1]}"
        )
    B, S, N = samples.shape
    ranks = (samples < targets.unsqueeze(1)).sum(dim=1) / S  # [B, N], in [0, 1]
    return ranks


def generate_and_predict(
        solver: ZstackSolver,
        batch_size: int,
        num_samples: int,
        chunk_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        device: Optional[str | torch.device] = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """
    Generates a batch of synthetic images + their aberration labels, then
    runs inference on the images and samples the predicted posteriors.
    """
    solver = solver.eval()
    if device is not None:
        solver = solver.to(device)
    with torch.inference_mode():
        example_data = solver.simulator.create_examples(
            batch_size,
            generator=generator,
            chunk_size=chunk_size,
        )
        images, coef_blocks = example_data[0], example_data[1:]  # [B, Z, H, W], [[B, N], ...]
        pred_coef_blocks = solver.predict_samples(images, num_samples)  # [[B, S, N], ...]
    return coef_blocks, pred_coef_blocks


def test_rank_uniformity(ranks: np.ndarray | torch.Tensor, num_bins: int) -> np.ndarray:
    """
    Tests rank-order statistics against a uniform distribution.

    Arguments:
    - ranks: [batch_size, num_variables] normalized to [0, 1]
    - num_bins: number of bins
    Returns:
    - [num_variables,] pvalues for rejecting the uniformity null hypothesis
    """
    if isinstance(ranks, torch.Tensor):
        ranks = ranks.numpy(force=True)
    num_vars = ranks.shape[1]
    bin_ids = np.minimum((ranks * num_bins).astype(np.int32), num_bins - 1)
    offsets = bin_ids + num_bins * np.arange(num_vars)
    counts = np.bincount(offsets.ravel(), minlength=num_vars * num_bins)
    counts = counts.reshape(num_vars, num_bins)
    result = stats.chisquare(counts, axis=1)
    pvalues = stats.false_discovery_control(result.pvalue)
    return pvalues
