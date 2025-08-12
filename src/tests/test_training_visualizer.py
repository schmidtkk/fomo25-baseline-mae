#!/usr/bin/env python3
"""
Unit tests for training visualization functionality.
Tests TrainingVisualizer and LiveTrainingMonitor classes.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock matplotlib to avoid display issues in testing
with patch.dict('sys.modules', {
    'matplotlib': MagicMock(),
    'matplotlib.pyplot': MagicMock(),
    'seaborn': MagicMock()
}):
    from utils.training_visualizer import TrainingVisualizer, LiveTrainingMonitor


class TestTrainingVisualizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test visualizer with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.save_dir = Path(self.temp_dir)
        self.visualizer = TrainingVisualizer(save_dir=self.save_dir)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization of TrainingVisualizer"""
        self.assertEqual(self.visualizer.save_dir, self.save_dir)
        self.assertEqual(len(self.visualizer.metrics_history), 0)
        self.assertEqual(len(self.visualizer.aggregation_history), 0)
        self.assertTrue(self.save_dir.exists())
    
    def test_update_metrics_basic(self):
        """Test basic metric updates"""
        epoch = 5
        train_metrics = {'loss': 0.5, 'accuracy': 0.8}
        val_metrics = {'loss': 0.4, 'accuracy': 0.85}
        
        self.visualizer.update_metrics(epoch, train_metrics, val_metrics)
        
        # Check metrics are stored
        self.assertEqual(len(self.visualizer.metrics_history), 1)
        stored = self.visualizer.metrics_history[0]
        
        self.assertEqual(stored['epoch'], epoch)
        self.assertEqual(stored['train_loss'], 0.5)
        self.assertEqual(stored['val_loss'], 0.4)
        self.assertEqual(stored['train_accuracy'], 0.8)
        self.assertEqual(stored['val_accuracy'], 0.85)
    
    def test_update_metrics_with_aggregation(self):
        """Test metric updates with aggregation methods"""
        epoch = 3
        train_metrics = {'loss': 0.3}
        val_metrics = {'loss': 0.25}
        aggregation_aurocs = {
            'mean_prob': 0.75,
            'mean_logit': 0.78,
            'noisy_or': 0.82
        }
        
        self.visualizer.update_metrics(epoch, train_metrics, val_metrics, aggregation_aurocs)
        
        # Check aggregation history
        self.assertEqual(len(self.visualizer.aggregation_history), 1)
        stored = self.visualizer.aggregation_history[0]
        
        self.assertEqual(stored['epoch'], epoch)
        self.assertEqual(stored['mean_prob'], 0.75)
        self.assertEqual(stored['mean_logit'], 0.78)
        self.assertEqual(stored['noisy_or'], 0.82)
    
    def test_multiple_metric_updates(self):
        """Test multiple metric updates across epochs"""
        # Add metrics for multiple epochs
        for epoch in range(5):
            train_metrics = {'loss': 0.5 - epoch * 0.05}  # Decreasing loss
            val_metrics = {'loss': 0.4 - epoch * 0.04}
            self.visualizer.update_metrics(epoch, train_metrics, val_metrics)
        
        # Check history length
        self.assertEqual(len(self.visualizer.metrics_history), 5)
        
        # Check progression
        first_epoch = self.visualizer.metrics_history[0]
        last_epoch = self.visualizer.metrics_history[-1]
        
        self.assertGreater(first_epoch['train_loss'], last_epoch['train_loss'])
        self.assertGreater(first_epoch['val_loss'], last_epoch['val_loss'])
    
    @patch('utils.training_visualizer.plt')
    @patch('utils.training_visualizer.sns')
    def test_generate_training_dashboard(self, mock_sns, mock_plt):
        """Test training dashboard generation"""
        # Add some sample data
        for epoch in range(10):
            train_metrics = {
                'loss': 0.5 - epoch * 0.03,
                'accuracy': 0.6 + epoch * 0.02
            }
            val_metrics = {
                'loss': 0.4 - epoch * 0.025,
                'accuracy': 0.65 + epoch * 0.015
            }
            aggregation_aurocs = {
                'mean_prob': 0.6 + epoch * 0.02,
                'mean_logit': 0.62 + epoch * 0.02,
                'noisy_or': 0.58 + epoch * 0.025
            }
            self.visualizer.update_metrics(epoch, train_metrics, val_metrics, aggregation_aurocs)
        
        # Test dashboard generation
        try:
            self.visualizer.generate_training_dashboard()
        except Exception as e:
            self.fail(f"Dashboard generation failed: {e}")
        
        # Verify plotting calls were made
        self.assertTrue(mock_plt.subplots.called)
        self.assertTrue(mock_plt.savefig.called)
    
    @patch('utils.training_visualizer.plt')
    def test_plot_loss_curves(self, mock_plt):
        """Test loss curve plotting"""
        # Add sample data
        for epoch in range(5):
            train_metrics = {'loss': 1.0 - epoch * 0.1}
            val_metrics = {'loss': 0.9 - epoch * 0.08}
            self.visualizer.update_metrics(epoch, train_metrics, val_metrics)
        
        # Mock axes
        mock_ax = MagicMock()
        
        try:
            self.visualizer._plot_loss_curves(mock_ax)
        except Exception as e:
            self.fail(f"Loss curve plotting failed: {e}")
        
        # Verify plot method calls
        self.assertTrue(mock_ax.plot.called)
        self.assertTrue(mock_ax.set_xlabel.called)
        self.assertTrue(mock_ax.set_ylabel.called)
    
    @patch('utils.training_visualizer.plt')
    def test_plot_auroc_comparison(self, mock_plt):
        """Test AUROC comparison plotting"""
        # Add sample data with aggregation
        for epoch in range(5):
            aggregation_aurocs = {
                'mean_prob': 0.7 + epoch * 0.02,
                'mean_logit': 0.72 + epoch * 0.02,
                'noisy_or': 0.68 + epoch * 0.025
            }
            self.visualizer.update_metrics(epoch, {}, {}, aggregation_aurocs)
        
        # Mock axes
        mock_ax = MagicMock()
        
        try:
            self.visualizer._plot_auroc_comparison(mock_ax)
        except Exception as e:
            self.fail(f"AUROC comparison plotting failed: {e}")
        
        # Verify plot method calls
        self.assertTrue(mock_ax.plot.called)
    
    def test_save_path_creation(self):
        """Test that save paths are created correctly"""
        epoch = 5
        
        # Test latest dashboard path
        latest_path = self.visualizer._get_latest_dashboard_path()
        self.assertEqual(latest_path.name, 'latest_dashboard.png')
        self.assertEqual(latest_path.parent, self.save_dir)
        
        # Test epoch-specific path
        epoch_path = self.visualizer._get_epoch_dashboard_path(epoch)
        self.assertEqual(epoch_path.name, 'dashboard_epoch_0005.png')
        self.assertEqual(epoch_path.parent, self.save_dir)
    
    def test_empty_data_handling(self):
        """Test handling of empty or minimal data"""
        # No data scenario
        mock_ax = MagicMock()
        
        try:
            self.visualizer._plot_loss_curves(mock_ax)
            self.visualizer._plot_auroc_comparison(mock_ax)
        except Exception as e:
            self.fail(f"Empty data handling failed: {e}")
    
    def test_metric_extraction(self):
        """Test extraction of different metric types"""
        # Add diverse metrics
        train_metrics = {
            'loss': 0.5,
            'accuracy': 0.8,
            'f1_score': 0.75,
            'custom_metric': 0.9
        }
        val_metrics = {
            'loss': 0.45,
            'accuracy': 0.82,
            'f1_score': 0.78,
            'custom_metric': 0.88
        }
        
        self.visualizer.update_metrics(0, train_metrics, val_metrics)
        
        stored = self.visualizer.metrics_history[0]
        
        # Check all metrics are stored with proper prefixes
        self.assertEqual(stored['train_loss'], 0.5)
        self.assertEqual(stored['val_loss'], 0.45)
        self.assertEqual(stored['train_accuracy'], 0.8)
        self.assertEqual(stored['val_accuracy'], 0.82)
        self.assertEqual(stored['train_custom_metric'], 0.9)
        self.assertEqual(stored['val_custom_metric'], 0.88)


class TestLiveTrainingMonitor(unittest.TestCase):
    
    def setUp(self):
        """Set up test monitor"""
        self.temp_dir = tempfile.mkdtemp()
        self.save_dir = Path(self.temp_dir)
        self.monitor = LiveTrainingMonitor(
            save_dir=self.save_dir,
            update_frequency=2
        )
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization of LiveTrainingMonitor"""
        self.assertEqual(self.monitor.update_frequency, 2)
        self.assertIsNotNone(self.monitor.visualizer)
        self.assertEqual(self.monitor.visualizer.save_dir, self.save_dir)
    
    def test_should_update_frequency(self):
        """Test update frequency logic"""
        # Should update on frequency intervals
        self.assertTrue(self.monitor._should_update(0))  # First epoch
        self.assertFalse(self.monitor._should_update(1))  # Not frequency
        self.assertTrue(self.monitor._should_update(2))   # Frequency
        self.assertFalse(self.monitor._should_update(3))  # Not frequency
        self.assertTrue(self.monitor._should_update(4))   # Frequency
    
    def test_on_validation_epoch_end_update(self):
        """Test validation epoch end callback with update"""
        # Mock trainer and pl_module
        mock_trainer = Mock()
        mock_trainer.callback_metrics = {
            'train_loss': torch.tensor(0.5),
            'val_loss': torch.tensor(0.4),
            'val_auroc_subject_mean_prob': torch.tensor(0.75),
            'val_auroc_subject_mean_logit': torch.tensor(0.78)
        }
        mock_trainer.current_epoch = 2  # Should trigger update
        
        mock_pl_module = Mock()
        
        try:
            self.monitor.on_validation_epoch_end(mock_trainer, mock_pl_module)
        except Exception as e:
            self.fail(f"Validation epoch end callback failed: {e}")
        
        # Check that visualizer received data
        self.assertEqual(len(self.monitor.visualizer.metrics_history), 1)
    
    def test_on_validation_epoch_end_no_update(self):
        """Test validation epoch end callback without update"""
        mock_trainer = Mock()
        mock_trainer.current_epoch = 1  # Should not trigger update
        mock_pl_module = Mock()
        
        initial_history_length = len(self.monitor.visualizer.metrics_history)
        
        self.monitor.on_validation_epoch_end(mock_trainer, mock_pl_module)
        
        # Should not have added to history
        self.assertEqual(len(self.monitor.visualizer.metrics_history), initial_history_length)
    
    def test_extract_aggregation_aurocs(self):
        """Test extraction of aggregation AUROC metrics"""
        callback_metrics = {
            'train_loss': torch.tensor(0.5),
            'val_loss': torch.tensor(0.4),
            'val_auroc_subject_mean_prob': torch.tensor(0.75),
            'val_auroc_subject_mean_logit': torch.tensor(0.78),
            'val_auroc_subject_noisy_or': torch.tensor(0.72),
            'other_metric': torch.tensor(0.9)
        }
        
        aurocs = self.monitor._extract_aggregation_aurocs(callback_metrics)
        
        expected = {
            'mean_prob': 0.75,
            'mean_logit': 0.78,
            'noisy_or': 0.72
        }
        
        self.assertEqual(aurocs, expected)
    
    def test_extract_train_val_metrics(self):
        """Test extraction of train/val metrics"""
        callback_metrics = {
            'train_loss': torch.tensor(0.5),
            'train_accuracy': torch.tensor(0.8),
            'val_loss': torch.tensor(0.4),
            'val_accuracy': torch.tensor(0.85),
            'val_auroc_subject_mean_prob': torch.tensor(0.75),  # Should be excluded
            'other_metric': torch.tensor(0.9)
        }
        
        train_metrics, val_metrics = self.monitor._extract_train_val_metrics(callback_metrics)
        
        expected_train = {'loss': 0.5, 'accuracy': 0.8}
        expected_val = {'loss': 0.4, 'accuracy': 0.85}
        
        self.assertEqual(train_metrics, expected_train)
        self.assertEqual(val_metrics, expected_val)
    
    def test_metric_filtering(self):
        """Test that aggregation-specific metrics are properly filtered"""
        callback_metrics = {
            'val_auroc_subject': torch.tensor(0.8),
            'val_auroc_subject_mean_prob': torch.tensor(0.75),
            'val_auroc_subject_mean_logit': torch.tensor(0.78),
            'val_auroc_subject_noisy_or': torch.tensor(0.72),
            'val_loss': torch.tensor(0.4)
        }
        
        train_metrics, val_metrics = self.monitor._extract_train_val_metrics(callback_metrics)
        
        # Should include the base auroc_subject but not method-specific ones
        self.assertIn('auroc_subject', val_metrics)
        self.assertNotIn('auroc_subject_mean_prob', val_metrics)
        self.assertNotIn('auroc_subject_mean_logit', val_metrics)
        self.assertNotIn('auroc_subject_noisy_or', val_metrics)
    
    @patch('utils.training_visualizer.TrainingVisualizer.generate_training_dashboard')
    def test_dashboard_generation_call(self, mock_generate):
        """Test that dashboard generation is called when appropriate"""
        mock_trainer = Mock()
        mock_trainer.callback_metrics = {'val_loss': torch.tensor(0.4)}
        mock_trainer.current_epoch = 0  # Should trigger update
        mock_pl_module = Mock()
        
        self.monitor.on_validation_epoch_end(mock_trainer, mock_pl_module)
        
        # Verify dashboard generation was called
        self.assertTrue(mock_generate.called)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty metrics
        mock_trainer = Mock()
        mock_trainer.callback_metrics = {}
        mock_trainer.current_epoch = 0
        mock_pl_module = Mock()
        
        try:
            self.monitor.on_validation_epoch_end(mock_trainer, mock_pl_module)
        except Exception as e:
            self.fail(f"Empty metrics handling failed: {e}")
        
        # None values in metrics
        mock_trainer.callback_metrics = {
            'val_loss': None,
            'train_loss': torch.tensor(0.5)
        }
        
        try:
            self.monitor.on_validation_epoch_end(mock_trainer, mock_pl_module)
        except Exception as e:
            self.fail(f"None values handling failed: {e}")


class TestVisualizationIntegration(unittest.TestCase):
    """Integration tests for visualization components"""
    
    def test_full_training_simulation(self):
        """Test full training simulation with visualization"""
        temp_dir = tempfile.mkdtemp()
        save_dir = Path(temp_dir)
        
        try:
            # Create monitor
            monitor = LiveTrainingMonitor(save_dir=save_dir, update_frequency=1)
            
            # Simulate training epochs
            for epoch in range(5):
                mock_trainer = Mock()
                mock_trainer.current_epoch = epoch
                mock_trainer.callback_metrics = {
                    'train_loss': torch.tensor(1.0 - epoch * 0.1),
                    'val_loss': torch.tensor(0.9 - epoch * 0.08),
                    'val_auroc_subject_mean_prob': torch.tensor(0.6 + epoch * 0.05),
                    'val_auroc_subject_mean_logit': torch.tensor(0.62 + epoch * 0.05),
                }
                mock_pl_module = Mock()
                
                monitor.on_validation_epoch_end(mock_trainer, mock_pl_module)
            
            # Verify data accumulation
            self.assertEqual(len(monitor.visualizer.metrics_history), 5)
            self.assertEqual(len(monitor.visualizer.aggregation_history), 5)
            
            # Verify progression
            first = monitor.visualizer.metrics_history[0]
            last = monitor.visualizer.metrics_history[-1]
            self.assertGreater(first['train_loss'], last['train_loss'])
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
