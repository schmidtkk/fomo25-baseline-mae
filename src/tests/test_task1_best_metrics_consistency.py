import unittest
import tempfile
import os
from unittest.mock import MagicMock, patch
import torch
from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule

from utils.enhanced_callbacks import EnhancedModelCheckpoint


class MockTrainer:
    def __init__(self, current_epoch=0, global_step=0, logged_metrics=None):
        self.current_epoch = current_epoch
        self.global_step = global_step
        self.logged_metrics = logged_metrics or {}
        self.callback_metrics = {}


class MockLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        
    def training_step(self, batch, batch_idx):
        return torch.tensor(0.5)
    
    def validation_step(self, batch, batch_idx):
        return torch.tensor(0.3)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class TestTask1BestMetricsConsistency(unittest.TestCase):
    """
    Test suite to validate that best metrics are tracked consistently
    and updated correctly throughout training.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoints_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_best_metrics_initialization(self):
        """Test that best metrics are initialized correctly."""
        callback = EnhancedModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            dirpath=self.checkpoints_dir,
            filename="best"
        )
        
        # Initialize manually since setup might not be called
        callback.best_metrics_file = os.path.join(self.checkpoints_dir, 'best_metrics.txt')
        callback._initialize_best_metrics_file()
        
        # Check that best metrics are initialized with correct modes
        expected_metrics = ['val/loss', 'val/auroc_subject', 'val/accuracy', 'val/mae', 'val/mse']
        for metric in expected_metrics:
            self.assertIn(metric, callback.best_metrics)
            self.assertIn('mode', callback.best_metrics[metric])
            self.assertIn('value', callback.best_metrics[metric])
            self.assertIn('epoch', callback.best_metrics[metric])
            self.assertIn('step', callback.best_metrics[metric])
    
    def test_best_metrics_update_logic(self):
        """Test that best metrics are updated correctly with improving values."""
        callback = EnhancedModelCheckpoint(
            monitor="val/loss", 
            mode="min",
            save_top_k=1,
            dirpath=self.checkpoints_dir,
            filename="best"
        )
        
        # Initialize
        callback.best_metrics_file = os.path.join(self.checkpoints_dir, 'best_metrics.txt')
        callback._initialize_best_metrics_file()
        
        # Test epoch 0: initial metrics
        trainer_epoch0 = MockTrainer(
            current_epoch=0, 
            global_step=100,
            logged_metrics={
                'val/loss': torch.tensor(0.8),
                'val/auroc_subject': torch.tensor(0.7),
                'val/accuracy': torch.tensor(0.6)
            }
        )
        
        current_metrics = callback._get_current_metrics(trainer_epoch0)
        any_new_best = callback._update_best_metrics(current_metrics, trainer_epoch0)
        
        # Should be new best for all metrics at epoch 0
        self.assertTrue(any_new_best)
        self.assertAlmostEqual(callback.best_metrics['val/loss']['value'], 0.8, places=5)
        self.assertEqual(callback.best_metrics['val/loss']['epoch'], 0)
        self.assertAlmostEqual(callback.best_metrics['val/auroc_subject']['value'], 0.7, places=5)
        self.assertAlmostEqual(callback.best_metrics['val/accuracy']['value'], 0.6, places=5)
        
        # Test epoch 10: improved metrics
        trainer_epoch10 = MockTrainer(
            current_epoch=10,
            global_step=1100, 
            logged_metrics={
                'val/loss': torch.tensor(0.7),  # Better (lower)
                'val/auroc_subject': torch.tensor(0.9),  # Better (higher) 
                'val/accuracy': torch.tensor(0.8)  # Better (higher)
            }
        )
        
        current_metrics = callback._get_current_metrics(trainer_epoch10)
        any_new_best = callback._update_best_metrics(current_metrics, trainer_epoch10)
        
        # Should detect improvements
        self.assertTrue(any_new_best)
        self.assertAlmostEqual(callback.best_metrics['val/loss']['value'], 0.7, places=5)
        self.assertEqual(callback.best_metrics['val/loss']['epoch'], 10)
        self.assertAlmostEqual(callback.best_metrics['val/auroc_subject']['value'], 0.9, places=5)
        self.assertEqual(callback.best_metrics['val/auroc_subject']['epoch'], 10)
        
        # Test epoch 20: worse metrics  
        trainer_epoch20 = MockTrainer(
            current_epoch=20,
            global_step=2100,
            logged_metrics={
                'val/loss': torch.tensor(0.8),  # Worse (higher)
                'val/auroc_subject': torch.tensor(0.8),  # Worse (lower)
                'val/accuracy': torch.tensor(0.7)  # Worse (lower)
            }
        )
        
        current_metrics = callback._get_current_metrics(trainer_epoch20)
        any_new_best = callback._update_best_metrics(current_metrics, trainer_epoch20)
        
        # Should not detect improvements
        self.assertFalse(any_new_best)
        # Best metrics should remain from epoch 10
        self.assertAlmostEqual(callback.best_metrics['val/loss']['value'], 0.7, places=5)
        self.assertEqual(callback.best_metrics['val/loss']['epoch'], 10)
        self.assertAlmostEqual(callback.best_metrics['val/auroc_subject']['value'], 0.9, places=5)
        self.assertEqual(callback.best_metrics['val/auroc_subject']['epoch'], 10)
    
    def test_best_metrics_file_persistence(self):
        """Test that best metrics are saved to and loaded from file correctly."""
        callback = EnhancedModelCheckpoint(
            monitor="val/loss",
            mode="min", 
            save_top_k=1,
            dirpath=self.checkpoints_dir,
            filename="best"
        )
        
        # Initialize 
        callback.best_metrics_file = os.path.join(self.checkpoints_dir, 'best_metrics.txt')
        callback._initialize_best_metrics_file()
        
        # Update some metrics
        trainer = MockTrainer(
            current_epoch=5,
            global_step=500,
            logged_metrics={
                'val/loss': torch.tensor(0.6),
                'val/auroc_subject': torch.tensor(0.85)
            }
        )
        
        current_metrics = callback._get_current_metrics(trainer)
        callback._update_best_metrics(current_metrics, trainer)
        
        # Verify file exists and contains correct data
        self.assertTrue(os.path.exists(callback.best_metrics_file))
        
        with open(callback.best_metrics_file, 'r') as f:
            content = f.read()
            self.assertIn('Best val/loss: 0.600000 (epoch 5, step 500)', content)
            self.assertIn('Best val/auroc_subject: 0.850000 (epoch 5, step 500)', content)
        
        # Test loading from file
        callback2 = EnhancedModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1, 
            dirpath=self.checkpoints_dir,
            filename="best2"
        )
        
        callback2.best_metrics_file = callback.best_metrics_file
        callback2._load_best_metrics_from_file()
        
        # Should load the same values
        self.assertAlmostEqual(callback2.best_metrics['val/loss']['value'], 0.6, places=5)
        self.assertEqual(callback2.best_metrics['val/loss']['epoch'], 5)
        self.assertAlmostEqual(callback2.best_metrics['val/auroc_subject']['value'], 0.85, places=5)
        self.assertEqual(callback2.best_metrics['val/auroc_subject']['epoch'], 5)
    
    def test_metric_mode_detection(self):
        """Test that metrics are compared using correct mode (min vs max)."""
        callback = EnhancedModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            dirpath=self.checkpoints_dir,
            filename="best"
        )
        
        callback.best_metrics_file = os.path.join(self.checkpoints_dir, 'best_metrics.txt')
        callback._initialize_best_metrics_file()
        
        # Verify modes are set correctly
        self.assertEqual(callback.best_metrics['val/loss']['mode'], 'min')
        self.assertEqual(callback.best_metrics['val/auroc_subject']['mode'], 'max')
        self.assertEqual(callback.best_metrics['val/accuracy']['mode'], 'max')
        self.assertEqual(callback.best_metrics['val/mae']['mode'], 'min')
        self.assertEqual(callback.best_metrics['val/mse']['mode'], 'min')
    
    def test_integration_with_checkpoint_saving(self):
        """Test that best metrics are updated when metrics improve."""
        callback = EnhancedModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            dirpath=self.checkpoints_dir,
            filename="best"
        )
        
        callback.best_metrics_file = os.path.join(self.checkpoints_dir, 'best_metrics.txt')
        callback._initialize_best_metrics_file()
        
        # Set initial best to worse value to ensure improvement is detected
        callback.best_metric_value = 0.8
        
        # Mock trainer with improving loss
        trainer = MockTrainer(
            current_epoch=10,
            global_step=1000,
            logged_metrics={
                'val/loss': torch.tensor(0.5),  # Better than initial 0.8
                'val/auroc_subject': torch.tensor(0.9)
            }
        )
        
        # Test the core logic without actually saving checkpoint
        current_metrics = callback._get_current_metrics(trainer)
        any_new_best = callback._update_best_metrics(current_metrics, trainer)
        
        # Verify best metrics were updated
        self.assertTrue(any_new_best)
        self.assertAlmostEqual(callback.best_metrics['val/loss']['value'], 0.5, places=5)
        self.assertEqual(callback.best_metrics['val/loss']['epoch'], 10)
        self.assertAlmostEqual(callback.best_metrics['val/auroc_subject']['value'], 0.9, places=5)
        
        # Verify best metrics file was created/updated
        self.assertTrue(os.path.exists(callback.best_metrics_file))
    
    def test_on_validation_end_updates_metrics(self):
        """Test that on_validation_end updates best metrics every epoch."""
        callback = EnhancedModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            dirpath=self.checkpoints_dir,
            filename="best"
        )
        
        callback.best_metrics_file = os.path.join(self.checkpoints_dir, 'best_metrics.txt')
        callback._initialize_best_metrics_file()
        
        # Create a mock Lightning module
        mock_module = MockLightningModule()
        
        # Simulate epoch 0 with initial metrics
        trainer_epoch0 = MockTrainer(
            current_epoch=0,
            global_step=100,
            logged_metrics={
                'val/loss': torch.tensor(0.8),
                'val/auroc_subject': torch.tensor(0.7)
            }
        )
        
        # Call on_validation_end (this is the fix we implemented)
        callback.on_validation_end(trainer_epoch0, mock_module)
        
        # Verify metrics were updated
        self.assertAlmostEqual(callback.best_metrics['val/loss']['value'], 0.8, places=5)
        self.assertEqual(callback.best_metrics['val/loss']['epoch'], 0)
        
        # Simulate epoch 10 with improved metrics (but no checkpoint saving)
        trainer_epoch10 = MockTrainer(
            current_epoch=10,
            global_step=1000,
            logged_metrics={
                'val/loss': torch.tensor(0.6),  # Better loss
                'val/auroc_subject': torch.tensor(0.85)  # Better AUROC
            }
        )
        
        # Call on_validation_end again
        callback.on_validation_end(trainer_epoch10, mock_module)
        
        # Verify metrics were updated to new best values
        self.assertAlmostEqual(callback.best_metrics['val/loss']['value'], 0.6, places=5)
        self.assertEqual(callback.best_metrics['val/loss']['epoch'], 10)
        self.assertAlmostEqual(callback.best_metrics['val/auroc_subject']['value'], 0.85, places=5)
        self.assertEqual(callback.best_metrics['val/auroc_subject']['epoch'], 10)
        
        # Verify the best metrics file was updated
        self.assertTrue(os.path.exists(callback.best_metrics_file))
        with open(callback.best_metrics_file, 'r') as f:
            content = f.read()
            self.assertIn('Best val/loss: 0.600000 (epoch 10, step 1000)', content)
            self.assertIn('Best val/auroc_subject: 0.850000 (epoch 10, step 1000)', content)


if __name__ == '__main__':
    unittest.main()
