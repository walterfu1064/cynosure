import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm


class NoValidationBar(TQDMProgressBar):
    """Dumb hack to avoid a visiual glitch where Jupyter appends a new validation bar each epoch"""
    def init_validation_tqdm(self):
        return Tqdm(disable=True)
