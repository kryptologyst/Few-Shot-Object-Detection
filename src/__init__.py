"""Main package initialization."""

__version__ = "1.0.0"
__author__ = "Few-Shot Object Detection Team"
__email__ = "contact@example.com"

from .models import create_model, MetaRCNN, FSRW, TFA, FewShotLoss
from .data import create_dataloaders, FewShotDataset, FewShotDataLoader
from .train import FewShotTrainer
from .eval import FewShotEvaluator
from .utils import set_seed, get_device, AverageMeter, EarlyStopping

__all__ = [
    "create_model",
    "MetaRCNN", 
    "FSRW", 
    "TFA", 
    "FewShotLoss",
    "create_dataloaders",
    "FewShotDataset",
    "FewShotDataLoader",
    "FewShotTrainer",
    "FewShotEvaluator",
    "set_seed",
    "get_device",
    "AverageMeter",
    "EarlyStopping"
]
