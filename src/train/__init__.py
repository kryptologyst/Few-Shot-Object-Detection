"""Training pipeline for few-shot object detection."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import time
from tqdm import tqdm
import json

from ..utils import AverageMeter, EarlyStopping, save_checkpoint, load_checkpoint, get_device
from ..models import FewShotLoss
from ..data import FewShotDataLoader

logger = logging.getLogger(__name__)


class FewShotTrainer:
    """Trainer for few-shot object detection models."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], 
                 device: Optional[torch.device] = None):
        """Initialize trainer.
        
        Args:
            model: Few-shot detection model.
            config: Training configuration.
            device: Device to train on.
        """
        self.model = model
        self.config = config
        self.device = device or get_device()
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = FewShotLoss(
            num_classes=config["num_classes"],
            lambda_proto=config.get("lambda_proto", 0.5),
            lambda_roi=config.get("lambda_roi", 0.5),
            temperature=config.get("temperature", 1.0)
        )
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize logging
        self.writer = SummaryWriter(config.get("log_dir", "logs"))
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 10),
            min_delta=config.get("min_delta", 0.001)
        )
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_name = self.config.get("optimizer", "adam")
        learning_rate = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        if optimizer_name.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_name = self.config.get("scheduler", None)
        
        if scheduler_name is None:
            return None
        elif scheduler_name.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("epochs", 100)
            )
        elif scheduler_name.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get("step_size", 30),
                gamma=self.config.get("gamma", 0.1)
            )
        elif scheduler_name.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train_epoch(self, train_loader: FewShotDataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Dict containing training metrics.
        """
        self.model.train()
        
        # Initialize meters
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, episode in enumerate(pbar):
            # Move data to device
            support_images = episode["support_images"].to(self.device)
            support_labels = episode["support_labels"].to(self.device)
            query_images = episode["query_images"].to(self.device)
            query_labels = episode["query_labels"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(support_images, support_images, support_labels)
            
            # Compute loss
            losses = self.criterion(outputs, query_labels)
            total_loss = losses["total_loss"]
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["grad_clip"]
                )
            
            self.optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                if "logits" in outputs:
                    predictions = torch.argmax(outputs["logits"], dim=1)
                elif "roi_logits" in outputs:
                    predictions = torch.argmax(outputs["roi_logits"], dim=1)
                else:
                    predictions = torch.argmax(outputs["proto_logits"], dim=1)
                
                accuracy = (predictions == query_labels).float().mean().item()
            
            # Update meters
            loss_meter.update(total_loss.item())
            accuracy_meter.update(accuracy)
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss_meter.avg:.4f}",
                "Acc": f"{accuracy_meter.avg:.4f}"
            })
            
            # Log to tensorboard
            if batch_idx % self.config.get("log_interval", 10) == 0:
                self.writer.add_scalar(
                    "Train/Loss", total_loss.item(),
                    self.current_epoch * len(train_loader) + batch_idx
                )
                self.writer.add_scalar(
                    "Train/Accuracy", accuracy,
                    self.current_epoch * len(train_loader) + batch_idx
                )
        
        return {
            "loss": loss_meter.avg,
            "accuracy": accuracy_meter.avg
        }
    
    def validate(self, val_loader: FewShotDataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Dict containing validation metrics.
        """
        self.model.eval()
        
        # Initialize meters
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        
        with torch.no_grad():
            for episode in tqdm(val_loader, desc="Validation"):
                # Move data to device
                support_images = episode["support_images"].to(self.device)
                support_labels = episode["support_labels"].to(self.device)
                query_images = episode["query_images"].to(self.device)
                query_labels = episode["query_labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(support_images, support_images, support_labels)
                
                # Compute loss
                losses = self.criterion(outputs, query_labels)
                total_loss = losses["total_loss"]
                
                # Compute accuracy
                if "logits" in outputs:
                    predictions = torch.argmax(outputs["logits"], dim=1)
                elif "roi_logits" in outputs:
                    predictions = torch.argmax(outputs["roi_logits"], dim=1)
                else:
                    predictions = torch.argmax(outputs["proto_logits"], dim=1)
                
                accuracy = (predictions == query_labels).float().mean().item()
                
                # Update meters
                loss_meter.update(total_loss.item())
                accuracy_meter.update(accuracy)
        
        return {
            "loss": loss_meter.avg,
            "accuracy": accuracy_meter.avg
        }
    
    def train(self, train_loader: FewShotDataLoader, 
              val_loader: FewShotDataLoader) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            
        Returns:
            Dict containing training history.
        """
        num_epochs = self.config.get("epochs", 100)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["accuracy"])
                else:
                    self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch:3d}/{num_epochs}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Log to tensorboard
            self.writer.add_scalar("Epoch/Train_Loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("Epoch/Train_Accuracy", train_metrics["accuracy"], epoch)
            self.writer.add_scalar("Epoch/Val_Loss", val_metrics["loss"], epoch)
            self.writer.add_scalar("Epoch/Val_Accuracy", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("Epoch/Learning_Rate", 
                                 self.optimizer.param_groups[0]["lr"], epoch)
            
            # Store metrics
            self.train_losses.append(train_metrics["loss"])
            self.val_metrics.append(val_metrics["accuracy"])
            
            # Save checkpoint
            is_best = val_metrics["accuracy"] > self.best_metric
            if is_best:
                self.best_metric = val_metrics["accuracy"]
            
            if epoch % self.config.get("save_interval", 10) == 0 or is_best:
                checkpoint_path = Path(self.config.get("checkpoint_dir", "checkpoints"))
                checkpoint_path.mkdir(exist_ok=True)
                
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_metrics["loss"],
                    val_metrics,
                    checkpoint_path / f"checkpoint_epoch_{epoch}.pth",
                    is_best
                )
            
            # Early stopping
            if self.early_stopping(val_metrics["loss"], self.model):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        final_path = Path(self.config.get("checkpoint_dir", "checkpoints")) / "final_model.pth"
        save_checkpoint(
            self.model,
            self.optimizer,
            self.current_epoch,
            train_metrics["loss"],
            val_metrics,
            final_path,
            is_best=False
        )
        
        self.writer.close()
        
        return {
            "train_losses": self.train_losses,
            "val_metrics": self.val_metrics,
            "best_metric": self.best_metric
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = load_checkpoint(self.model, self.optimizer, checkpoint_path, self.device)
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["metrics"].get("accuracy", 0.0)
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


class FewShotEvaluator:
    """Evaluator for few-shot object detection models."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initialize evaluator.
        
        Args:
            model: Few-shot detection model.
            device: Device to evaluate on.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, test_loader: FewShotDataLoader) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            test_loader: Test data loader.
            
        Returns:
            Dict containing evaluation metrics.
        """
        self.model.eval()
        
        # Initialize meters
        accuracy_meter = AverageMeter()
        per_class_accuracy = {}
        
        with torch.no_grad():
            for episode in tqdm(test_loader, desc="Evaluation"):
                # Move data to device
                support_images = episode["support_images"].to(self.device)
                support_labels = episode["support_labels"].to(self.device)
                query_images = episode["query_images"].to(self.device)
                query_labels = episode["query_labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(support_images, support_images, support_labels)
                
                # Get predictions
                if "logits" in outputs:
                    predictions = torch.argmax(outputs["logits"], dim=1)
                elif "roi_logits" in outputs:
                    predictions = torch.argmax(outputs["roi_logits"], dim=1)
                else:
                    predictions = torch.argmax(outputs["proto_logits"], dim=1)
                
                # Compute overall accuracy
                accuracy = (predictions == query_labels).float().mean().item()
                accuracy_meter.update(accuracy)
                
                # Compute per-class accuracy
                for label in torch.unique(query_labels):
                    mask = query_labels == label
                    if mask.sum() > 0:
                        class_acc = (predictions[mask] == query_labels[mask]).float().mean().item()
                        if label.item() not in per_class_accuracy:
                            per_class_accuracy[label.item()] = []
                        per_class_accuracy[label.item()].append(class_acc)
        
        # Compute average per-class accuracy
        avg_per_class_acc = {}
        for class_id, accs in per_class_accuracy.items():
            avg_per_class_acc[f"class_{class_id}_accuracy"] = np.mean(accs)
        
        metrics = {
            "overall_accuracy": accuracy_meter.avg,
            **avg_per_class_acc
        }
        
        return metrics
    
    def evaluate_episodes(self, test_loader: FewShotDataLoader, 
                         num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate over multiple episodes.
        
        Args:
            test_loader: Test data loader.
            num_episodes: Number of episodes to evaluate.
            
        Returns:
            Dict containing evaluation metrics.
        """
        all_accuracies = []
        
        for episode_idx, episode in enumerate(test_loader):
            if episode_idx >= num_episodes:
                break
                
            # Move data to device
            support_images = episode["support_images"].to(self.device)
            support_labels = episode["support_labels"].to(self.device)
            query_images = episode["query_images"].to(self.device)
            query_labels = episode["query_labels"].to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(support_images, support_images, support_labels)
                
                # Get predictions
                if "logits" in outputs:
                    predictions = torch.argmax(outputs["logits"], dim=1)
                elif "roi_logits" in outputs:
                    predictions = torch.argmax(outputs["roi_logits"], dim=1)
                else:
                    predictions = torch.argmax(outputs["proto_logits"], dim=1)
                
                # Compute accuracy for this episode
                accuracy = (predictions == query_labels).float().mean().item()
                all_accuracies.append(accuracy)
        
        # Compute statistics
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        confidence_interval = 1.96 * std_accuracy / np.sqrt(len(all_accuracies))
        
        return {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "confidence_interval": confidence_interval,
            "min_accuracy": np.min(all_accuracies),
            "max_accuracy": np.max(all_accuracies)
        }
