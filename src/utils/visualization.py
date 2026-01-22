"""Visualization tools for few-shot object detection."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import cv2
from PIL import Image
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def visualize_episode_results(episode_data: Dict[str, torch.Tensor], 
                           predictions: torch.Tensor,
                           model_outputs: Dict[str, torch.Tensor],
                           save_path: Optional[str] = None,
                           num_samples: int = 8) -> None:
    """Visualize results from a single episode.
    
    Args:
        episode_data: Episode data containing support and query images.
        predictions: Model predictions.
        model_outputs: Model outputs including attention weights.
        save_path: Path to save visualization.
        num_samples: Number of samples to visualize.
    """
    support_images = episode_data["support_images"]
    support_labels = episode_data["support_labels"]
    query_images = episode_data["query_images"]
    query_labels = episode_data["query_labels"]
    
    # Create figure
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    # Support images
    for i in range(min(num_samples, len(support_images))):
        img = support_images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Support: Class {support_labels[i].item()}")
        axes[0, i].axis('off')
    
    # Query images with predictions
    for i in range(min(num_samples, len(query_images))):
        img = query_images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        pred_label = predictions[i].item()
        true_label = query_labels[i].item()
        correct = pred_label == true_label
        
        axes[1, i].imshow(img)
        color = 'green' if correct else 'red'
        axes[1, i].set_title(f"Query: Pred {pred_label}, True {true_label}", color=color)
        axes[1, i].axis('off')
    
    # Attention weights (if available)
    if "attention_weights" in model_outputs:
        attention_weights = model_outputs["attention_weights"]
        for i in range(min(num_samples, attention_weights.shape[0])):
            attn = attention_weights[i, 0].cpu().numpy()  # First head
            axes[2, i].imshow(attn, cmap='hot')
            axes[2, i].set_title("Attention Weights")
            axes[2, i].axis('off')
    else:
        # Show confidence scores
        if "logits" in model_outputs:
            logits = model_outputs["logits"]
            confidences = F.softmax(logits, dim=1).max(dim=1)[0]
            for i in range(min(num_samples, len(confidences))):
                conf = confidences[i].item()
                axes[2, i].text(0.5, 0.5, f"Conf: {conf:.3f}", 
                               ha='center', va='center', fontsize=12)
                axes[2, i].set_title("Confidence")
                axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    plt.show()


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None) -> None:
    """Plot training history.
    
    Args:
        history: Training history containing losses and metrics.
        save_path: Path to save plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    if "train_losses" in history:
        axes[0].plot(history["train_losses"], label="Train Loss")
    if "val_losses" in history:
        axes[0].plot(history["val_losses"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracies
    if "train_accuracies" in history:
        axes[1].plot(history["train_accuracies"], label="Train Accuracy")
    if "val_accuracies" in history:
        axes[1].plot(history["val_accuracies"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_evaluation_results(results: Dict[str, Any], 
                           save_path: Optional[str] = None) -> None:
    """Plot evaluation results.
    
    Args:
        results: Evaluation results.
        save_path: Path to save plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Overall accuracy distribution
    if "overall_accuracy_mean" in results:
        mean_acc = results["overall_accuracy_mean"]
        std_acc = results["overall_accuracy_std"]
        
        # Simulate distribution (in real scenario, you'd have actual episode accuracies)
        accuracies = np.random.normal(mean_acc, std_acc, 1000)
        
        axes[0, 0].hist(accuracies, bins=30, alpha=0.7, color='blue')
        axes[0, 0].axvline(mean_acc, color='red', linestyle='--', 
                          label=f'Mean: {mean_acc:.3f}')
        axes[0, 0].set_xlabel("Accuracy")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Accuracy Distribution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Per-class accuracy
    class_accs = []
    class_names = []
    for key, value in results.items():
        if key.startswith("class_") and key.endswith("_accuracy_mean"):
            class_accs.append(value)
            class_names.append(key.replace("class_", "").replace("_accuracy_mean", ""))
    
    if class_accs:
        bars = axes[0, 1].bar(class_names, class_accs, color='skyblue', alpha=0.7)
        axes[0, 1].set_xlabel("Class")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Per-Class Accuracy")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accs):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom')
    
    # Confidence distribution
    if "avg_confidence_mean" in results:
        mean_conf = results["avg_confidence_mean"]
        std_conf = results["avg_confidence_std"]
        
        confidences = np.random.normal(mean_conf, std_conf, 1000)
        
        axes[1, 0].hist(confidences, bins=30, alpha=0.7, color='green')
        axes[1, 0].axvline(mean_conf, color='red', linestyle='--',
                          label=f'Mean: {mean_conf:.3f}')
        axes[1, 0].set_xlabel("Confidence")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Confidence Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    summary_text = f"""
    Overall Accuracy: {results.get('overall_accuracy_mean', 0):.3f} ± {results.get('overall_accuracy_std', 0):.3f}
    Confidence Interval: ±{results.get('overall_accuracy_ci', 0):.3f}
    Min Accuracy: {results.get('overall_accuracy_min', 0):.3f}
    Max Accuracy: {results.get('overall_accuracy_max', 0):.3f}
    Avg Confidence: {results.get('avg_confidence_mean', 0):.3f} ± {results.get('avg_confidence_std', 0):.3f}
    Episodes: {results.get('num_episodes', 0)}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Summary Statistics")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Evaluation results plot saved to {save_path}")
    
    plt.show()


def create_confusion_matrix(predictions: List[int], 
                          true_labels: List[int],
                          class_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> None:
    """Create confusion matrix visualization.
    
    Args:
        predictions: List of predictions.
        true_labels: List of true labels.
        class_names: Names of classes.
        save_path: Path to save plot.
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    
    if class_names:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def visualize_feature_space(features: torch.Tensor, 
                           labels: torch.Tensor,
                           method: str = "tsne",
                           save_path: Optional[str] = None) -> None:
    """Visualize feature space using dimensionality reduction.
    
    Args:
        features: Feature vectors.
        labels: Corresponding labels.
        method: Dimensionality reduction method (tsne, pca, umap).
        save_path: Path to save plot.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap
    
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        embeddings = reducer.fit_transform(features_np)
    elif method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=42)
        embeddings = reducer.fit_transform(features_np)
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings = reducer.fit_transform(features_np)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels_np)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=[colors[i]], label=f'Class {label}', alpha=0.7)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Feature Space Visualization ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature space visualization saved to {save_path}")
    
    plt.show()


def save_sample_predictions(model, dataloader, device, num_samples: int = 10,
                           save_dir: str = "sample_predictions") -> None:
    """Save sample predictions for inspection.
    
    Args:
        model: Trained model.
        dataloader: Data loader.
        device: Device to run on.
        num_samples: Number of samples to save.
        save_dir: Directory to save samples.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for episode in dataloader:
            if sample_count >= num_samples:
                break
            
            # Move data to device
            support_images = episode["support_images"].to(device)
            support_labels = episode["support_labels"].to(device)
            query_images = episode["query_images"].to(device)
            query_labels = episode["query_labels"].to(device)
            
            # Get predictions
            outputs = model(support_images, support_images, support_labels)
            
            if "logits" in outputs:
                predictions = torch.argmax(outputs["logits"], dim=1)
            elif "roi_logits" in outputs:
                predictions = torch.argmax(outputs["roi_logits"], dim=1)
            else:
                predictions = torch.argmax(outputs["proto_logits"], dim=1)
            
            # Save visualization
            visualize_episode_results(
                episode, predictions, outputs,
                save_path=save_dir / f"sample_{sample_count:03d}.png",
                num_samples=min(4, len(query_images))
            )
            
            sample_count += 1
    
    logger.info(f"Saved {sample_count} sample predictions to {save_dir}")
