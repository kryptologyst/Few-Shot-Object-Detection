"""Core utilities for few-shot object detection."""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> str:
    """Get model size in MB.
    
    Args:
        model: PyTorch model.
        
    Returns:
        str: Model size in MB.
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return f"{size_all_mb:.2f} MB"


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        """Reset all values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """Update with new value.
        
        Args:
            val: New value.
            n: Number of samples.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: Union[str, Path],
    is_best: bool = False
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        loss: Current loss.
        metrics: Dictionary of metrics.
        filepath: Path to save checkpoint.
        is_best: Whether this is the best model.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'is_best': is_best
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: Union[str, Path],
    device: torch.device
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer (optional).
        filepath: Path to checkpoint.
        device: Device to load on.
        
    Returns:
        Dict containing checkpoint information.
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint


def create_directory(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path.
        
    Returns:
        Path: Created directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets.
    
    Args:
        labels: List of class labels.
        num_classes: Number of classes.
        
    Returns:
        torch.Tensor: Class weights.
    """
    from collections import Counter
    
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total_samples / (num_classes * class_counts[i])
        else:
            weight = 1.0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            model: PyTorch model.
            
        Returns:
            bool: True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()
