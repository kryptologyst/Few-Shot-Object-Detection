#!/usr/bin/env python3
"""Main training script for few-shot object detection."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import set_seed, get_device, create_directory
from src.models import create_model
from src.data import create_dataloaders, create_toy_dataset
from src.train import FewShotTrainer
from src.eval import FewShotEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Set seed for reproducibility
    set_seed(cfg.device.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directories
    output_dir = create_directory(cfg.paths.output_root)
    checkpoint_dir = create_directory(cfg.paths.checkpoint_root)
    log_dir = create_directory(cfg.paths.log_root)
    
    # Create toy dataset if needed
    if cfg.data.dataset_name == "toy":
        toy_data_dir = Path(cfg.data.data_root) / "toy_dataset"
        if not toy_data_dir.exists():
            logger.info("Creating toy dataset...")
            create_toy_dataset(
                str(toy_data_dir),
                num_classes=cfg.model.num_classes,
                samples_per_class=200
            )
        cfg.data.data_root = str(toy_data_dir)
        cfg.data.dataset_name = "imagefolder"
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(OmegaConf.to_container(cfg, resolve=True))
    
    # Create model
    logger.info(f"Creating model: {cfg.model.name}")
    model = create_model(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        support_shot=cfg.model.support_shot,
        backbone=cfg.model.backbone,
        feature_dim=cfg.model.feature_dim,
        attention_dim=cfg.model.get("attention_dim", 128)
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer_config = OmegaConf.to_container(cfg, resolve=True)
    trainer_config["log_dir"] = str(log_dir)
    trainer_config["checkpoint_dir"] = str(checkpoint_dir)
    
    trainer = FewShotTrainer(model, trainer_config, device)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = FewShotEvaluator(model, device)
    
    # Basic evaluation
    results = evaluator.evaluate(val_loader, cfg.evaluation.num_episodes)
    
    # Per-class evaluation
    class_results = evaluator.evaluate_class_performance(val_loader, cfg.evaluation.num_episodes)
    results["per_class"] = class_results
    
    # Save results
    results_dir = create_directory(output_dir / "results")
    evaluator.save_results(results, results_dir / "final_results.json")
    
    # Print summary
    evaluator.print_summary(results)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
