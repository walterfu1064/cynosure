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
        chunk_size: Optional[int] = None,
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
    - chunk_size: if given, split the batch dimension into chunks to reduce peak memory
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
    chunk_size = chunk_size or B

    ranks = []
    for start in range(0, B, chunk_size):
        stop = min(start + chunk_size, B)
        targets_chunk = targets[start:stop]
        samples_chunk = samples[start:stop]
        ranks_chunk = (samples_chunk < targets_chunk.unsqueeze(1)).sum(dim=1) / S
        ranks.append(ranks_chunk)
    ranks = torch.cat(ranks, dim=0)  # [B, N], in [0, 1]
    return ranks


def generate_and_predict(
        solver: ZstackSolver,
        batch_size: int,
        num_samples: int,
        chunk_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        device: Optional[str | torch.device] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Generates a batch of synthetic images + their aberration labels, then
    runs inference on the images and samples the predicted posteriors.
    """
    solver = solver.eval()
    if device is not None:
        solver = solver.to(device)

    chunk_size = chunk_size or batch_size
    coef_blocks = []
    pred_coef_blocks = []
    with torch.inference_mode():
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            images, z, *label_blocks = solver.simulator.create_examples(
                stop - start,
                generator=generator,
            )
            coef_blocks_chunk = [cb.cpu() for cb in label_blocks]  # [[chunk, N], ...]
            pred_coef_blocks_chunk = solver.predict_samples(images, num_samples, z=z)  # [[chunk, S, N], ...]
            pred_coef_blocks_chunk = [pcb.cpu() for pcb in pred_coef_blocks_chunk]
            coef_blocks.append(coef_blocks_chunk)
            pred_coef_blocks.append(pred_coef_blocks_chunk)
    coef_blocks = [torch.cat(cb, dim=0) for cb in zip(*coef_blocks)]  # [[B, N], ...]
    pred_coef_blocks = [torch.cat(pcb, dim=0) for pcb in zip(*pred_coef_blocks)]  # [[B, S, N], ...]
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
