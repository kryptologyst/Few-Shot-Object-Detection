"""Evaluation pipeline for few-shot object detection."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
from collections import defaultdict

from ..utils import get_device
from ..data import FewShotDataLoader

logger = logging.getLogger(__name__)


class FewShotEvaluator:
    """Comprehensive evaluator for few-shot object detection models."""
    
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
    
    def evaluate_single_episode(self, episode: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate a single episode.
        
        Args:
            episode: Episode data containing support and query samples.
            
        Returns:
            Dict containing episode metrics.
        """
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
                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=1)
            elif "roi_logits" in outputs:
                logits = outputs["roi_logits"]
                predictions = torch.argmax(logits, dim=1)
            else:
                logits = outputs["proto_logits"]
                predictions = torch.argmax(logits, dim=1)
            
            # Compute accuracy
            accuracy = (predictions == query_labels).float().mean().item()
            
            # Compute confidence scores
            confidence_scores = torch.softmax(logits, dim=1).max(dim=1)[0]
            avg_confidence = confidence_scores.mean().item()
            
            # Compute per-class metrics
            unique_labels = torch.unique(query_labels)
            per_class_acc = {}
            
            for label in unique_labels:
                mask = query_labels == label
                if mask.sum() > 0:
                    class_acc = (predictions[mask] == query_labels[mask]).float().mean().item()
                    per_class_acc[f"class_{label.item()}_accuracy"] = class_acc
        
        return {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            **per_class_acc
        }
    
    def evaluate(self, test_loader: FewShotDataLoader, 
                num_episodes: Optional[int] = None) -> Dict[str, float]:
        """Evaluate the model over multiple episodes.
        
        Args:
            test_loader: Test data loader.
            num_episodes: Number of episodes to evaluate (None for all).
            
        Returns:
            Dict containing comprehensive evaluation metrics.
        """
        logger.info("Starting evaluation...")
        
        all_accuracies = []
        all_confidences = []
        per_class_accuracies = defaultdict(list)
        episode_times = []
        
        episode_count = 0
        
        for episode in tqdm(test_loader, desc="Evaluation"):
            if num_episodes is not None and episode_count >= num_episodes:
                break
            
            start_time = time.time()
            
            # Evaluate episode
            episode_metrics = self.evaluate_single_episode(episode)
            
            episode_time = time.time() - start_time
            episode_times.append(episode_time)
            
            # Collect metrics
            all_accuracies.append(episode_metrics["accuracy"])
            all_confidences.append(episode_metrics["avg_confidence"])
            
            # Collect per-class accuracies
            for key, value in episode_metrics.items():
                if key.startswith("class_") and key.endswith("_accuracy"):
                    per_class_accuracies[key].append(value)
            
            episode_count += 1
        
        # Compute overall statistics
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        confidence_interval = 1.96 * std_accuracy / np.sqrt(len(all_accuracies))
        
        mean_confidence = np.mean(all_confidences)
        std_confidence = np.std(all_confidences)
        
        avg_episode_time = np.mean(episode_times)
        
        # Compute per-class statistics
        per_class_stats = {}
        for class_key, accs in per_class_accuracies.items():
            if accs:  # Only if we have data for this class
                per_class_stats[f"{class_key}_mean"] = np.mean(accs)
                per_class_stats[f"{class_key}_std"] = np.std(accs)
                per_class_stats[f"{class_key}_count"] = len(accs)
        
        # Compile final metrics
        metrics = {
            "num_episodes": episode_count,
            "overall_accuracy_mean": mean_accuracy,
            "overall_accuracy_std": std_accuracy,
            "overall_accuracy_ci": confidence_interval,
            "overall_accuracy_min": np.min(all_accuracies),
            "overall_accuracy_max": np.max(all_accuracies),
            "avg_confidence_mean": mean_confidence,
            "avg_confidence_std": std_confidence,
            "avg_episode_time": avg_episode_time,
            "total_evaluation_time": sum(episode_times),
            **per_class_stats
        }
        
        logger.info(f"Evaluation completed: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        return metrics
    
    def evaluate_few_shot_performance(self, test_loader: FewShotDataLoader,
                                    support_shots: List[int] = [1, 5, 10],
                                    num_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate performance with different support shot numbers.
        
        Args:
            test_loader: Test data loader.
            support_shots: List of support shot numbers to evaluate.
            num_episodes: Number of episodes per support shot.
            
        Returns:
            Dict containing performance for each support shot.
        """
        logger.info(f"Evaluating few-shot performance with support shots: {support_shots}")
        
        results = {}
        
        for support_shot in support_shots:
            logger.info(f"Evaluating with {support_shot} support shots...")
            
            # Modify test loader for this support shot
            original_support_shot = test_loader.support_shot
            test_loader.support_shot = support_shot
            
            # Evaluate
            metrics = self.evaluate(test_loader, num_episodes)
            results[f"{support_shot}_shot"] = metrics
            
            # Restore original support shot
            test_loader.support_shot = original_support_shot
        
        return results
    
    def evaluate_class_performance(self, test_loader: FewShotDataLoader,
                                 num_episodes: int = 200) -> Dict[str, Any]:
        """Evaluate performance per class.
        
        Args:
            test_loader: Test data loader.
            num_episodes: Number of episodes to evaluate.
            
        Returns:
            Dict containing detailed class performance.
        """
        logger.info("Evaluating per-class performance...")
        
        class_accuracies = defaultdict(list)
        class_confidences = defaultdict(list)
        episode_count = 0
        
        for episode in tqdm(test_loader, desc="Class Evaluation"):
            if episode_count >= num_episodes:
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
                    logits = outputs["logits"]
                    predictions = torch.argmax(logits, dim=1)
                elif "roi_logits" in outputs:
                    logits = outputs["roi_logits"]
                    predictions = torch.argmax(logits, dim=1)
                else:
                    logits = outputs["proto_logits"]
                    predictions = torch.argmax(logits, dim=1)
                
                confidence_scores = torch.softmax(logits, dim=1).max(dim=1)[0]
            
            # Compute per-class metrics
            unique_labels = torch.unique(query_labels)
            for label in unique_labels:
                mask = query_labels == label
                if mask.sum() > 0:
                    class_acc = (predictions[mask] == query_labels[mask]).float().mean().item()
                    class_conf = confidence_scores[mask].mean().item()
                    
                    class_accuracies[label.item()].append(class_acc)
                    class_confidences[label.item()].append(class_conf)
            
            episode_count += 1
        
        # Compute statistics per class
        class_stats = {}
        for class_id in class_accuracies:
            accs = class_accuracies[class_id]
            confs = class_confidences[class_id]
            
            class_stats[f"class_{class_id}"] = {
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs),
                "accuracy_count": len(accs),
                "confidence_mean": np.mean(confs),
                "confidence_std": np.std(confs)
            }
        
        return class_stats
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results.
            output_path: Path to save results.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print evaluation summary.
        
        Args:
            results: Evaluation results.
        """
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if "overall_accuracy_mean" in results:
            print(f"Overall Accuracy: {results['overall_accuracy_mean']:.4f} ± {results['overall_accuracy_std']:.4f}")
            print(f"Confidence Interval (95%): ±{results['overall_accuracy_ci']:.4f}")
            print(f"Min Accuracy: {results['overall_accuracy_min']:.4f}")
            print(f"Max Accuracy: {results['overall_accuracy_max']:.4f}")
            print(f"Average Confidence: {results['avg_confidence_mean']:.4f} ± {results['avg_confidence_std']:.4f}")
            print(f"Average Episode Time: {results['avg_episode_time']:.4f}s")
            print(f"Total Episodes: {results['num_episodes']}")
        
        # Print per-class results if available
        class_keys = [k for k in results.keys() if k.startswith("class_") and k.endswith("_accuracy_mean")]
        if class_keys:
            print("\nPer-Class Accuracy:")
            for key in sorted(class_keys):
                class_id = key.replace("class_", "").replace("_accuracy_mean", "")
                acc_mean = results[key]
                acc_std = results.get(key.replace("_mean", "_std"), 0.0)
                print(f"  Class {class_id}: {acc_mean:.4f} ± {acc_std:.4f}")
        
        print("="*50)


def evaluate_model(model: nn.Module, test_loader: FewShotDataLoader,
                  output_dir: str = "results", num_episodes: int = 100) -> Dict[str, Any]:
    """Convenience function to evaluate a model.
    
    Args:
        model: Few-shot detection model.
        test_loader: Test data loader.
        output_dir: Output directory for results.
        num_episodes: Number of episodes to evaluate.
        
    Returns:
        Dict containing evaluation results.
    """
    evaluator = FewShotEvaluator(model)
    
    # Basic evaluation
    results = evaluator.evaluate(test_loader, num_episodes)
    
    # Per-class evaluation
    class_results = evaluator.evaluate_class_performance(test_loader, num_episodes)
    results["per_class"] = class_results
    
    # Save results
    output_path = Path(output_dir) / "evaluation_results.json"
    evaluator.save_results(results, output_path)
    
    # Print summary
    evaluator.print_summary(results)
    
    return results
