from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
import torch


class NoValidationBar(TQDMProgressBar):
    """Dumb hack to avoid a visiual glitch where Jupyter appends a new validation bar each epoch"""
    def init_validation_tqdm(self):
        return Tqdm(disable=True)


def safe_load_from_checkpoint(
        checkpoint_path: str | Path,
        model: pl.LightningModule,
        as_eval: bool = True,
) -> pl.LightningModule:
    """Safely loads weights from a checkpoint into an identically-shaped model"""
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    if as_eval:
        model = model.eval()
    return model


def load_logged_metrics(
        trainer: Optional[pl.Trainer] = None,
        metrics_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Returns the metric history of a run.

    Looks for metrics in the trainer first. If that fails (e.g.,
    because the model was loaded from a checkpoint), the metrics
    path is used instead.
    """
    if trainer is not None and trainer.logged_metrics:
        log_path = Path(trainer.logger.log_dir) / "metrics.csv"
        if log_path.is_file():
            return pd.read_csv(log_path)

    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        if metrics_path.is_file():
            return pd.read_csv(metrics_path)

    raise FileNotFoundError(f"No metrics found in trainer or at {metrics_path}")
