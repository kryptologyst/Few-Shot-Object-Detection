#!/usr/bin/env python3
"""Evaluation script for few-shot object detection."""

import argparse
import torch
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import create_model
from src.data import create_dataloaders, create_toy_dataset
from src.eval import FewShotEvaluator
from src.utils import get_device, load_checkpoint
from omegaconf import OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate few-shot object detection model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--support_shots", type=str, default="5", help="Support shots to evaluate (comma-separated)")
    parser.add_argument("--per_class", action="store_true", help="Evaluate per-class performance")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create toy dataset if needed
    if config.data.dataset_name == "toy":
        toy_data_dir = Path(config.data.data_root) / "toy_dataset"
        if not toy_data_dir.exists():
            logger.info("Creating toy dataset...")
            create_toy_dataset(
                str(toy_data_dir),
                num_classes=config.model.num_classes,
                samples_per_class=200
            )
        config.data.data_root = str(toy_data_dir)
        config.data.dataset_name = "imagefolder"
    
    # Create data loaders
    logger.info("Creating data loaders...")
    _, test_loader = create_dataloaders(OmegaConf.to_container(config, resolve=True))
    
    # Create model
    logger.info(f"Creating model: {config.model.name}")
    model = create_model(
        model_name=config.model.name,
        num_classes=config.model.num_classes,
        support_shot=config.model.support_shot,
        backbone=config.model.backbone,
        feature_dim=config.model.feature_dim,
        attention_dim=config.model.get("attention_dim", 128)
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(model, None, args.checkpoint, device)
    
    # Create evaluator
    evaluator = FewShotEvaluator(model, device)
    
    # Parse support shots
    support_shots = [int(x.strip()) for x in args.support_shots.split(",")]
    
    # Evaluate with different support shots
    if len(support_shots) > 1:
        logger.info(f"Evaluating with support shots: {support_shots}")
        results = evaluator.evaluate_few_shot_performance(
            test_loader, support_shots, args.num_episodes
        )
    else:
        # Single support shot evaluation
        test_loader.support_shot = support_shots[0]
        results = evaluator.evaluate(test_loader, args.num_episodes)
    
    # Per-class evaluation
    if args.per_class:
        logger.info("Evaluating per-class performance...")
        class_results = evaluator.evaluate_class_performance(test_loader, args.num_episodes)
        results["per_class"] = class_results
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_results(results, output_dir / "evaluation_results.json")
    
    # Print summary
    evaluator.print_summary(results)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
