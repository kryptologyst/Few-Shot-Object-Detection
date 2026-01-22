"""Unit tests for few-shot object detection models."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import MetaRCNN, FSRW, TFA, FewShotLoss, create_model
from src.data import CIFARFewShotDataset, ImageFolderFewShotDataset, FewShotDataLoader, create_toy_dataset
from src.utils import set_seed, get_device, AverageMeter, EarlyStopping
from src.train import FewShotTrainer
from src.eval import FewShotEvaluator


class TestModels:
    """Test model implementations."""
    
    def test_meta_rcnn_creation(self):
        """Test Meta-RCNN model creation."""
        model = MetaRCNN(num_classes=5, support_shot=5)
        assert model.num_classes == 5
        assert model.support_shot == 5
        assert isinstance(model.backbone, nn.Module)
        assert isinstance(model.roi_head, nn.Module)
        assert isinstance(model.proto_net, nn.Module)
    
    def test_fsrw_creation(self):
        """Test FSRW model creation."""
        model = FSRW(num_classes=5, support_shot=5)
        assert model.num_classes == 5
        assert model.support_shot == 5
        assert isinstance(model.backbone, nn.Module)
        assert isinstance(model.attention, nn.Module)
        assert isinstance(model.classifier, nn.Module)
    
    def test_tfa_creation(self):
        """Test TFA model creation."""
        model = TFA(num_classes=5, support_shot=5)
        assert model.num_classes == 5
        assert model.support_shot == 5
        assert isinstance(model.backbone, nn.Module)
        assert isinstance(model.classifier, nn.Module)
        assert isinstance(model.feature_adaptation, nn.Module)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        batch_size = 2
        num_support = 10
        num_classes = 5
        
        # Create dummy data
        query_images = torch.randn(batch_size, 3, 224, 224)
        support_images = torch.randn(num_support, 3, 224, 224)
        support_labels = torch.randint(0, num_classes, (num_support,))
        
        # Test Meta-RCNN
        model = MetaRCNN(num_classes=num_classes, support_shot=5)
        outputs = model(query_images, support_images, support_labels)
        
        assert "proto_logits" in outputs
        assert "roi_logits" in outputs
        assert outputs["proto_logits"].shape == (batch_size, num_classes)
        assert outputs["roi_logits"].shape == (batch_size, num_classes)
        
        # Test FSRW
        model = FSRW(num_classes=num_classes, support_shot=5)
        outputs = model(query_images, support_images, support_labels)
        
        assert "logits" in outputs
        assert "attention_weights" in outputs
        assert outputs["logits"].shape == (batch_size, num_classes)
        
        # Test TFA
        model = TFA(num_classes=num_classes, support_shot=5)
        outputs = model(query_images, support_images, support_labels)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, num_classes)
    
    def test_create_model_function(self):
        """Test model creation function."""
        # Test valid models
        for model_name in ["meta_rcnn", "fsrw", "tfa"]:
            model = create_model(model_name, num_classes=5, support_shot=5)
            assert isinstance(model, nn.Module)
        
        # Test invalid model
        with pytest.raises(ValueError):
            create_model("invalid_model", num_classes=5, support_shot=5)


class TestLossFunctions:
    """Test loss function implementations."""
    
    def test_few_shot_loss(self):
        """Test few-shot loss computation."""
        num_classes = 5
        batch_size = 4
        
        # Create dummy outputs and labels
        outputs = {
            "proto_logits": torch.randn(batch_size, num_classes),
            "roi_logits": torch.randn(batch_size, num_classes)
        }
        query_labels = torch.randint(0, num_classes, (batch_size,))
        
        # Test loss computation
        criterion = FewShotLoss(num_classes=num_classes)
        losses = criterion(outputs, query_labels)
        
        assert "proto_loss" in losses
        assert "roi_loss" in losses
        assert "total_loss" in losses
        assert losses["total_loss"] > 0
        
        # Test loss weights
        criterion = FewShotLoss(num_classes=num_classes, lambda_proto=0.8, lambda_roi=0.2)
        losses = criterion(outputs, query_labels)
        
        expected_total = 0.8 * losses["proto_loss"] + 0.2 * losses["roi_loss"]
        assert abs(losses["total_loss"] - expected_total) < 1e-6


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_cifar_dataset_creation(self):
        """Test CIFAR dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = CIFARFewShotDataset(
                root=temp_dir,
                split="train",
                dataset_name="cifar10",
                num_classes=5,
                support_shot=5,
                query_shot=10
            )
            
            assert len(dataset) > 0
            assert len(dataset.classes) == 5
            
            # Test data loading
            image, label = dataset[0]
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, int)
            assert image.shape == (3, 224, 224)
            assert 0 <= label < 5
    
    def test_toy_dataset_creation(self):
        """Test toy dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            create_toy_dataset(
                output_dir=temp_dir,
                num_classes=3,
                samples_per_class=20
            )
            
            # Check if dataset was created
            dataset_dir = Path(temp_dir)
            assert dataset_dir.exists()
            
            for class_id in range(3):
                class_dir = dataset_dir / f"class_{class_id}"
                assert class_dir.exists()
                assert len(list(class_dir.glob("*.png"))) == 20
    
    def test_few_shot_dataloader(self):
        """Test few-shot data loader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create toy dataset
            create_toy_dataset(
                output_dir=temp_dir,
                num_classes=3,
                samples_per_class=50
            )
            
            # Create dataset
            dataset = ImageFolderFewShotDataset(
                root=temp_dir,
                num_classes=3,
                support_shot=5,
                query_shot=10
            )
            
            # Create data loader
            dataloader = FewShotDataLoader(
                dataset,
                batch_size=1,
                num_episodes=5,
                support_shot=5,
                query_shot=10
            )
            
            # Test data loading
            episode_count = 0
            for episode in dataloader:
                assert "support_images" in episode
                assert "support_labels" in episode
                assert "query_images" in episode
                assert "query_labels" in episode
                
                assert episode["support_images"].shape[0] == episode["support_labels"].shape[0]
                assert episode["query_images"].shape[0] == episode["query_labels"].shape[0]
                
                episode_count += 1
                if episode_count >= 3:  # Test first few episodes
                    break


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # Set seed again
        set_seed(42)
        
        # Generate same random numbers
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_average_meter(self):
        """Test AverageMeter functionality."""
        meter = AverageMeter()
        
        # Test reset
        meter.reset()
        assert meter.val == 0.0
        assert meter.avg == 0.0
        assert meter.sum == 0.0
        assert meter.count == 0
        
        # Test update
        meter.update(1.0, 1)
        assert meter.val == 1.0
        assert meter.avg == 1.0
        assert meter.sum == 1.0
        assert meter.count == 1
        
        meter.update(2.0, 2)
        assert meter.val == 2.0
        assert meter.avg == 1.5  # (1.0 + 2.0*2) / 3
        assert meter.sum == 5.0
        assert meter.count == 3
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        model = nn.Linear(10, 1)
        early_stopping = EarlyStopping(patience=3, min_delta=0.1)
        
        # Test improvement
        assert not early_stopping(1.0, model)
        assert not early_stopping(0.8, model)  # Improvement
        assert not early_stopping(0.9, model)  # No improvement
        assert not early_stopping(0.9, model)  # No improvement
        assert not early_stopping(0.9, model)  # No improvement
        assert early_stopping(0.9, model)  # Should stop now


class TestTraining:
    """Test training functionality."""
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        model = MetaRCNN(num_classes=5, support_shot=5)
        config = {
            "num_classes": 5,
            "learning_rate": 1e-4,
            "optimizer": "adam",
            "scheduler": None,
            "epochs": 10,
            "log_dir": "logs",
            "checkpoint_dir": "checkpoints"
        }
        
        trainer = FewShotTrainer(model, config)
        
        assert trainer.model == model
        assert trainer.config == config
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)
        assert isinstance(trainer.criterion, FewShotLoss)
    
    def test_evaluator_creation(self):
        """Test evaluator creation."""
        model = MetaRCNN(num_classes=5, support_shot=5)
        evaluator = FewShotEvaluator(model)
        
        assert evaluator.model == model
        assert evaluator.device.type in ["cuda", "mps", "cpu"]


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training process."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create toy dataset
            create_toy_dataset(
                output_dir=temp_dir,
                num_classes=3,
                samples_per_class=100
            )
            
            # Create model
            model = MetaRCNN(num_classes=3, support_shot=5)
            
            # Create dataset
            dataset = ImageFolderFewShotDataset(
                root=temp_dir,
                num_classes=3,
                support_shot=5,
                query_shot=10
            )
            
            # Create data loader
            dataloader = FewShotDataLoader(
                dataset,
                batch_size=1,
                num_episodes=5,
                support_shot=5,
                query_shot=10
            )
            
            # Create trainer
            config = {
                "num_classes": 3,
                "learning_rate": 1e-3,
                "optimizer": "adam",
                "scheduler": None,
                "epochs": 2,
                "log_dir": temp_dir,
                "checkpoint_dir": temp_dir,
                "lambda_proto": 0.5,
                "lambda_roi": 0.5,
                "temperature": 1.0
            }
            
            trainer = FewShotTrainer(model, config)
            
            # Train for one epoch
            train_metrics = trainer.train_epoch(dataloader)
            
            assert "loss" in train_metrics
            assert "accuracy" in train_metrics
            assert train_metrics["loss"] > 0
            assert 0 <= train_metrics["accuracy"] <= 1
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create toy dataset
            create_toy_dataset(
                output_dir=temp_dir,
                num_classes=3,
                samples_per_class=50
            )
            
            # Create model
            model = MetaRCNN(num_classes=3, support_shot=5)
            
            # Create dataset
            dataset = ImageFolderFewShotDataset(
                root=temp_dir,
                num_classes=3,
                support_shot=5,
                query_shot=10
            )
            
            # Create data loader
            dataloader = FewShotDataLoader(
                dataset,
                batch_size=1,
                num_episodes=3,
                support_shot=5,
                query_shot=10
            )
            
            # Create evaluator
            evaluator = FewShotEvaluator(model)
            
            # Evaluate
            results = evaluator.evaluate(dataloader, num_episodes=3)
            
            assert "overall_accuracy_mean" in results
            assert "num_episodes" in results
            assert results["num_episodes"] == 3
            assert 0 <= results["overall_accuracy_mean"] <= 1


if __name__ == "__main__":
    pytest.main([__file__])
