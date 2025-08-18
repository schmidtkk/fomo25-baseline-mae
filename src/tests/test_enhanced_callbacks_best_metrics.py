#!/usr/bin/env python
"""
Test suite for enhanced callbacks best metrics tracking functionality.

Tests:
1. EnhancedModelCheckpoint best metrics file creation and updates
2. LossPlottingCallback best metrics annotations
3. Integration with different task types and monitor metrics
4. File I/O robustness and error handling
"""

import unittest
import tempfile
import shutil
import os
from unittest.mock import Mock, MagicMock, patch
import torch

# Import the classes we want to test
from utils.enhanced_callbacks import EnhancedModelCheckpoint, LossPlottingCallback


class TestEnhancedModelCheckpointBestMetrics(unittest.TestCase):
    """Test the best metrics tracking functionality in EnhancedModelCheckpoint."""
    
    def setUp(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_callback = EnhancedModelCheckpoint(
            dirpath=self.temp_dir,
            filename="test_best",
            monitor="val/loss",
            mode="min",
            save_top_k=1
        )
        
        # Mock trainer and module
        self.mock_trainer = Mock()
        self.mock_trainer.current_epoch = 10
        self.mock_trainer.global_step = 1000
        self.mock_trainer.logged_metrics = {}
        self.mock_trainer.callback_metrics = {}
        
        self.mock_module = Mock()
        
        # Initialize the callback
        self.checkpoint_callback.setup(self.mock_trainer, self.mock_module, "fit")
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_best_metrics_file_initialization(self):
        """Test that best metrics file is created and initialized properly."""
        expected_file = os.path.join(self.temp_dir, "best_metrics.txt")
        self.assertTrue(os.path.exists(expected_file))
        
        # Check file contains proper header
        with open(expected_file, 'r') as f:
            content = f.read()
            self.assertIn("# Best Metrics History", content)
            self.assertIn("# Updated:", content)
            
    def test_best_metrics_tracking_val_loss(self):
        """Test tracking of validation loss as best metric."""
        # Simulate validation loss improvement
        self.mock_trainer.logged_metrics = {
            'val/loss': torch.tensor(0.5),
            'val/auroc_subject': torch.tensor(0.75)
        }
        
        # Update best metrics
        current_metrics = self.checkpoint_callback._get_current_metrics(self.mock_trainer)
        any_new_best = self.checkpoint_callback._update_best_metrics(current_metrics, self.mock_trainer)
        
        # Check that val/loss was updated (should be better than initial inf)
        self.assertTrue(any_new_best)
        self.assertEqual(self.checkpoint_callback.best_metrics['val/loss']['value'], 0.5)
        self.assertEqual(self.checkpoint_callback.best_metrics['val/loss']['epoch'], 10)
        self.assertEqual(self.checkpoint_callback.best_metrics['val/loss']['step'], 1000)
        
    def test_best_metrics_tracking_val_auroc(self):
        """Test tracking of validation AUROC as best metric."""
        # Simulate AUROC improvement
        self.mock_trainer.logged_metrics = {
            'val/auroc_subject': torch.tensor(0.85),
            'val/loss': torch.tensor(0.3)
        }
        
        current_metrics = self.checkpoint_callback._get_current_metrics(self.mock_trainer)
        any_new_best = self.checkpoint_callback._update_best_metrics(current_metrics, self.mock_trainer)
        
        # Check that val/auroc_subject was updated (should be better than initial 0.0)
        self.assertTrue(any_new_best)
        self.assertEqual(self.checkpoint_callback.best_metrics['val/auroc_subject']['value'], 0.85)
        
    def test_best_metrics_no_improvement(self):
        """Test that no update occurs when metrics don't improve."""
        # Set an initial good val/loss
        self.checkpoint_callback.best_metrics['val/loss']['value'] = 0.2
        
        # Simulate worse validation loss
        self.mock_trainer.logged_metrics = {
            'val/loss': torch.tensor(0.5)  # Worse than 0.2
        }
        
        current_metrics = self.checkpoint_callback._get_current_metrics(self.mock_trainer)
        any_new_best = self.checkpoint_callback._update_best_metrics(current_metrics, self.mock_trainer)
        
        # Should return False (no new best)
        self.assertFalse(any_new_best)
        # Value should remain unchanged
        self.assertEqual(self.checkpoint_callback.best_metrics['val/loss']['value'], 0.2)
        
    def test_best_metrics_file_persistence(self):
        """Test that best metrics are saved to and loaded from file correctly."""
        # Set up some best metrics
        self.checkpoint_callback.best_metrics['val/loss']['value'] = 0.123456
        self.checkpoint_callback.best_metrics['val/loss']['epoch'] = 50
        self.checkpoint_callback.best_metrics['val/loss']['step'] = 2500
        
        self.checkpoint_callback.best_metrics['val/auroc_subject']['value'] = 0.876543
        self.checkpoint_callback.best_metrics['val/auroc_subject']['epoch'] = 75
        self.checkpoint_callback.best_metrics['val/auroc_subject']['step'] = 3750
        
        # Save to file
        self.checkpoint_callback._save_best_metrics_to_file()
        
        # Create new callback and load from file
        new_callback = EnhancedModelCheckpoint(
            dirpath=self.temp_dir,
            filename="test_best2",
            monitor="val/loss",
            mode="min"
        )
        new_callback.best_metrics_file = os.path.join(self.temp_dir, "best_metrics.txt")
        new_callback._load_best_metrics_from_file()
        
        # Check that metrics were loaded correctly
        self.assertAlmostEqual(new_callback.best_metrics['val/loss']['value'], 0.123456, places=6)
        self.assertEqual(new_callback.best_metrics['val/loss']['epoch'], 50)
        self.assertEqual(new_callback.best_metrics['val/loss']['step'], 2500)
        
        self.assertAlmostEqual(new_callback.best_metrics['val/auroc_subject']['value'], 0.876543, places=6)
        self.assertEqual(new_callback.best_metrics['val/auroc_subject']['epoch'], 75)
        
    def test_enhanced_terminal_output_format(self):
        """Test the enhanced terminal output includes best metrics comparison."""
        # Mock print function to capture output
        with patch('builtins.print') as mock_print:
            # Set up scenario where monitor metric improves
            self.checkpoint_callback.monitor = "val/loss"
            self.checkpoint_callback.mode = "min"
            self.checkpoint_callback.best_metric_value = None  # First time
            
            self.mock_trainer.logged_metrics = {
                'val/loss': torch.tensor(0.5),
                'val/auroc_subject': torch.tensor(0.75)
            }
            
            # Call _save_checkpoint (but don't actually save)
            with patch.object(self.checkpoint_callback, '_save_model') as mock_save:
                self.checkpoint_callback._save_checkpoint(self.mock_trainer, "test.ckpt")
                
            # Check that enhanced output was printed
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            output_str = '\n'.join(print_calls)
            
            self.assertIn("🏆 NEW BEST CHECKPOINT SAVED!", output_str)
            self.assertIn("⭐ NEW RECORD!", output_str)
            self.assertIn("📋 Current Metrics vs Best:", output_str)


class TestLossPlottingCallbackBestMetrics(unittest.TestCase):
    """Test the best metrics plotting functionality in LossPlottingCallback."""
    
    def setUp(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.best_metrics_file = os.path.join(self.temp_dir, "best_metrics.txt")
        
        # Create a sample best metrics file
        with open(self.best_metrics_file, 'w') as f:
            f.write("# Best Metrics History\n")
            f.write("# Updated: 2025-08-17 14:30:45\n")
            f.write("Best val/loss: 0.234567 (epoch 45, step 1234)\n")
            f.write("Best val/auroc_subject: 0.876543 (epoch 67, step 2345)\n")
        
        self.plotting_callback = LossPlottingCallback(
            save_dir=self.temp_dir,
            enable_plotting=True,
            best_metrics_file=self.best_metrics_file
        )
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_load_best_metrics_from_file(self):
        """Test loading best metrics from file for plotting."""
        best_metrics = self.plotting_callback._load_best_metrics()
        
        self.assertIn('val/loss', best_metrics)
        self.assertIn('val/auroc_subject', best_metrics)
        
        self.assertAlmostEqual(best_metrics['val/loss']['value'], 0.234567, places=6)
        self.assertEqual(best_metrics['val/loss']['epoch'], 45)
        
        self.assertAlmostEqual(best_metrics['val/auroc_subject']['value'], 0.876543, places=6)
        self.assertEqual(best_metrics['val/auroc_subject']['epoch'], 67)
        
    def test_load_best_metrics_file_not_exists(self):
        """Test graceful handling when best metrics file doesn't exist."""
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.txt")
        callback = LossPlottingCallback(
            save_dir=self.temp_dir,
            best_metrics_file=nonexistent_file
        )
        
        best_metrics = callback._load_best_metrics()
        self.assertEqual(best_metrics, {})
        
    def test_load_best_metrics_malformed_file(self):
        """Test graceful handling of malformed best metrics file."""
        malformed_file = os.path.join(self.temp_dir, "malformed.txt")
        with open(malformed_file, 'w') as f:
            f.write("# Header\n")
            f.write("Malformed line without proper format\n")
            f.write("Best val/loss: not_a_number (epoch invalid)\n")
            f.write("Best val/auroc: 0.5 (malformed info)\n")
        
        callback = LossPlottingCallback(
            save_dir=self.temp_dir,
            best_metrics_file=malformed_file
        )
        
        # Should not crash and return empty dict
        best_metrics = callback._load_best_metrics()
        self.assertIsInstance(best_metrics, dict)
        
    @patch('matplotlib.pyplot.axhline')
    @patch('matplotlib.pyplot.text')
    def test_add_best_metrics_annotation(self, mock_text, mock_axhline):
        """Test adding best metrics annotation to plot axis."""
        # Mock axis
        mock_ax = Mock()
        mock_ax.transAxes = Mock()
        mock_ax.get_title.return_value = "Original Title"
        
        # Test with val/loss metric
        self.plotting_callback._add_best_metrics_annotation(
            mock_ax, 'val/loss', [1, 2, 3], "Test Plot"
        )
        
        # Check that horizontal line was added
        mock_axhline.assert_called_once()
        # Check that text annotation was added
        mock_text.assert_called_once()
        # Check that title was updated
        mock_ax.set_title.assert_called_once()
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplots')
    def test_save_plots_with_best_metrics(self, mock_subplots, mock_savefig):
        """Test that plots are saved with best metrics annotations."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Set up some data for plotting
        self.plotting_callback.val_losses = [0.8, 0.6, 0.4, 0.3]
        self.plotting_callback.val_steps = [100, 200, 300, 400]
        
        # Mock the best metrics annotation method
        with patch.object(self.plotting_callback, '_add_best_metrics_annotation') as mock_annotate:
            self.plotting_callback._save_plots(10)
            
            # Check that annotation method was called for validation loss plot
            mock_annotate.assert_called()
            
        # Check that plot was saved
        mock_savefig.assert_called_once()


class TestIntegrationBestMetrics(unittest.TestCase):
    """Integration tests for best metrics tracking across callbacks."""
    
    def test_checkpoint_and_plotting_integration(self):
        """Test that checkpoint callback and plotting callback work together."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create checkpoint callback
            checkpoint_callback = EnhancedModelCheckpoint(
                dirpath=temp_dir,
                filename="best",
                monitor="val/loss",
                mode="min"
            )
            
            # Create plotting callback with same best metrics file
            plotting_callback = LossPlottingCallback(
                save_dir=temp_dir,
                best_metrics_file=os.path.join(temp_dir, "best_metrics.txt")
            )
            
            # Mock trainer
            mock_trainer = Mock()
            mock_trainer.current_epoch = 20
            mock_trainer.global_step = 2000
            mock_trainer.logged_metrics = {
                'val/loss': torch.tensor(0.25),
                'val/auroc_subject': torch.tensor(0.80)
            }
            
            mock_module = Mock()
            
            # Setup callbacks
            checkpoint_callback.setup(mock_trainer, mock_module, "fit")
            plotting_callback.setup(mock_trainer, mock_module, "fit")
            
            # Simulate metric updates in checkpoint callback
            current_metrics = checkpoint_callback._get_current_metrics(mock_trainer)
            checkpoint_callback._update_best_metrics(current_metrics, mock_trainer)
            
            # Check that plotting callback can read the same metrics
            best_metrics = plotting_callback._load_best_metrics()
            
            self.assertIn('val/loss', best_metrics)
            self.assertEqual(best_metrics['val/loss']['value'], 0.25)
            self.assertEqual(best_metrics['val/loss']['epoch'], 20)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
