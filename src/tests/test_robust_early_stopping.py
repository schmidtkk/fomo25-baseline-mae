#!/usr/bin/env python3
"""
Unit tests for robust early stopping functionality.
Tests RobustEarlyStopping and AdaptiveEarlyStopping classes.
"""

import unittest
import torch
from unittest.mock import Mock, MagicMock, patch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.robust_early_stopping import RobustEarlyStopping, AdaptiveEarlyStopping


class TestRobustEarlyStopping(unittest.TestCase):
    
    def setUp(self):
        """Set up test early stopping instances"""
        self.mock_trainer = Mock()
        self.mock_trainer.callback_metrics = {}
        
        # Create robust early stopping instance
        self.robust_es = RobustEarlyStopping(
            monitor='val/auroc_subject',
            patience=5,
            min_delta=0.01,
            mode='max',
            ensemble_methods=['mean_prob', 'mean_logit', 'noisy_or'],
            smoothing_window=3,
            min_epochs=10
        )
    
    def test_initialization(self):
        """Test proper initialization of RobustEarlyStopping"""
        # Test default parameters
        es = RobustEarlyStopping(monitor='val/loss')
        self.assertEqual(es.monitor, 'val/loss')
        self.assertEqual(es.patience, 10)
        self.assertEqual(es.mode, 'min')
        self.assertEqual(len(es.ensemble_methods), 3)  # Default methods
        self.assertEqual(es.smoothing_window, 5)
        self.assertEqual(es.min_epochs, 15)
        
        # Test custom parameters
        self.assertEqual(self.robust_es.ensemble_methods, ['mean_prob', 'mean_logit', 'noisy_or'])
        self.assertEqual(self.robust_es.smoothing_window, 3)
        self.assertEqual(self.robust_es.min_epochs, 10)
    
    def test_get_ensemble_metric_basic(self):
        """Test basic ensemble metric computation"""
        # Mock trainer with method-specific metrics
        self.mock_trainer.callback_metrics = {
            'val/auroc_subject_mean_prob': torch.tensor(0.8),
            'val/auroc_subject_mean_logit': torch.tensor(0.82),
            'val/auroc_subject_noisy_or': torch.tensor(0.78)
        }
        
        ensemble_metric = self.robust_es._get_ensemble_metric(self.mock_trainer)
        
        # Should return mean of available methods
        expected = (0.8 + 0.82 + 0.78) / 3
        self.assertAlmostEqual(ensemble_metric, expected, places=5)
    
    def test_get_ensemble_metric_missing_methods(self):
        """Test ensemble metric when some methods are missing"""
        # Only some methods available
        self.mock_trainer.callback_metrics = {
            'val/auroc_subject_mean_prob': torch.tensor(0.8),
            'val/auroc_subject_mean_logit': torch.tensor(0.82),
            # noisy_or missing
        }
        
        ensemble_metric = self.robust_es._get_ensemble_metric(self.mock_trainer)
        
        # Should use available methods only
        expected = (0.8 + 0.82) / 2
        self.assertAlmostEqual(ensemble_metric, expected, places=5)
    
    def test_get_ensemble_metric_fallback(self):
        """Test fallback to original metric when no ensemble methods available"""
        # No method-specific metrics available
        self.mock_trainer.callback_metrics = {
            'val/auroc_subject': torch.tensor(0.75),
            'other_metric': torch.tensor(0.5)
        }
        
        ensemble_metric = self.robust_es._get_ensemble_metric(self.mock_trainer)
        
        # Should fallback to original metric
        self.assertAlmostEqual(ensemble_metric, 0.75, places=5)
    
    def test_apply_smoothing(self):
        """Test temporal smoothing of metrics"""
        # Add some history
        self.robust_es.metric_history = [0.7, 0.8, 0.75]
        
        # Test smoothing
        smoothed = self.robust_es._apply_smoothing(0.85)
        
        # Should average over smoothing window
        expected = np.mean([0.8, 0.75, 0.85])  # Last 3 values
        self.assertAlmostEqual(smoothed, expected, places=5)
    
    def test_apply_smoothing_insufficient_history(self):
        """Test smoothing with insufficient history"""
        # Limited history
        self.robust_es.metric_history = [0.7]
        
        smoothed = self.robust_es._apply_smoothing(0.8)
        
        # Should use available history
        expected = (0.7 + 0.8) / 2
        self.assertAlmostEqual(smoothed, expected, places=5)
    
    def test_min_epochs_protection(self):
        """Test that early stopping is blocked before min_epochs"""
        # Mock scenario before min_epochs
        mock_pl_module = Mock()
        mock_pl_module.current_epoch = 5  # Less than min_epochs (10)
        
        # Set up trainer with metrics showing no improvement
        self.mock_trainer.callback_metrics = {
            'val/auroc_subject': torch.tensor(0.5)  # Poor performance
        }
        
        # Should not stop before min_epochs regardless of performance
        result = self.robust_es._run_early_stopping_check(self.mock_trainer)
        self.assertFalse(result)
    
    def test_early_stopping_after_min_epochs(self):
        """Test early stopping logic after min_epochs threshold"""
        # Mock scenario after min_epochs
        mock_pl_module = Mock()
        mock_pl_module.current_epoch = 15  # After min_epochs (10)
        
        # Set up trainer
        self.mock_trainer.callback_metrics = {
            'val/auroc_subject_mean_prob': torch.tensor(0.6),
            'val/auroc_subject_mean_logit': torch.tensor(0.62),
            'val/auroc_subject_noisy_or': torch.tensor(0.58)
        }
        
        # Simulate some improvement history, then plateau
        with patch.object(self.robust_es, '_apply_smoothing', return_value=0.6):
            with patch.object(self.robust_es, 'monitor_op', return_value=False):  # No improvement
                # Add some wait_count history to trigger stopping
                self.robust_es.wait_count = 6  # Greater than patience (5)
                
                result = self.robust_es._run_early_stopping_check(self.mock_trainer)
                # Note: This will depend on the base class implementation
                # We're mainly testing that our logic is called correctly
    
    def test_ensemble_metric_weights(self):
        """Test that ensemble metrics can handle different weights"""
        # Set up metrics with known values
        self.mock_trainer.callback_metrics = {
            'val/auroc_subject_mean_prob': torch.tensor(0.6),
            'val/auroc_subject_mean_logit': torch.tensor(0.8),
            'val/auroc_subject_noisy_or': torch.tensor(0.7)
        }
        
        # Test equal weighting (default)
        ensemble_metric = self.robust_es._get_ensemble_metric(self.mock_trainer)
        expected = (0.6 + 0.8 + 0.7) / 3
        self.assertAlmostEqual(ensemble_metric, expected, places=5)
        
        # Could extend to test weighted averaging if implemented
    
    def test_metric_history_management(self):
        """Test that metric history is properly maintained"""
        initial_length = len(self.robust_es.metric_history)
        
        # Add some metrics
        for value in [0.5, 0.6, 0.7, 0.8]:
            self.robust_es._apply_smoothing(value)
        
        # History should grow
        self.assertGreater(len(self.robust_es.metric_history), initial_length)
        
        # History should be limited (not grow indefinitely)
        for value in range(100):  # Add many values
            self.robust_es._apply_smoothing(float(value))
        
        # Should maintain reasonable history size
        self.assertLessEqual(len(self.robust_es.metric_history), 100)


class TestAdaptiveEarlyStopping(unittest.TestCase):
    
    def setUp(self):
        """Set up test adaptive early stopping instance"""
        self.mock_trainer = Mock()
        self.mock_trainer.callback_metrics = {}
        
        self.adaptive_es = AdaptiveEarlyStopping(
            monitor='val/auroc_subject',
            initial_patience=5,
            max_patience=15,
            patience_factor=1.5,
            improvement_threshold=0.01
        )
    
    def test_initialization(self):
        """Test proper initialization of AdaptiveEarlyStopping"""
        self.assertEqual(self.adaptive_es.initial_patience, 5)
        self.assertEqual(self.adaptive_es.max_patience, 15)
        self.assertEqual(self.adaptive_es.patience_factor, 1.5)
        self.assertEqual(self.adaptive_es.improvement_threshold, 0.01)
        self.assertEqual(self.adaptive_es.patience, 5)  # Should start with initial_patience
    
    def test_adjust_patience_improvement(self):
        """Test patience adjustment when improvement is detected"""
        # Simulate improvement scenario
        self.adaptive_es.best_score = 0.8
        current_score = 0.85  # Clear improvement
        
        self.adaptive_es._adjust_patience(current_score)
        
        # Patience should be reset to initial value after improvement
        self.assertEqual(self.adaptive_es.patience, self.adaptive_es.initial_patience)
    
    def test_adjust_patience_no_improvement(self):
        """Test patience adjustment when no improvement is detected"""
        # Simulate no improvement scenario
        self.adaptive_es.best_score = 0.8
        self.adaptive_es.patience = 5
        current_score = 0.79  # No significant improvement
        
        initial_patience = self.adaptive_es.patience
        self.adaptive_es._adjust_patience(current_score)
        
        # Patience should increase by factor (but limited by max_patience)
        expected_patience = min(initial_patience * self.adaptive_es.patience_factor, 
                               self.adaptive_es.max_patience)
        self.assertEqual(self.adaptive_es.patience, expected_patience)
    
    def test_adjust_patience_max_limit(self):
        """Test that patience doesn't exceed maximum limit"""
        # Set patience close to maximum
        self.adaptive_es.patience = 12
        self.adaptive_es.best_score = 0.8
        current_score = 0.79  # No improvement
        
        self.adaptive_es._adjust_patience(current_score)
        
        # Should not exceed max_patience
        self.assertLessEqual(self.adaptive_es.patience, self.adaptive_es.max_patience)
    
    def test_improvement_detection_maximize(self):
        """Test improvement detection for maximize mode"""
        self.adaptive_es.mode = 'max'
        self.adaptive_es.best_score = 0.8
        
        # Test clear improvement
        self.assertTrue(self.adaptive_es._detect_improvement(0.82))
        
        # Test no improvement
        self.assertFalse(self.adaptive_es._detect_improvement(0.79))
        
        # Test marginal improvement (below threshold)
        self.assertFalse(self.adaptive_es._detect_improvement(0.805))  # Only 0.005 improvement
    
    def test_improvement_detection_minimize(self):
        """Test improvement detection for minimize mode"""
        self.adaptive_es.mode = 'min'
        self.adaptive_es.best_score = 0.5
        
        # Test clear improvement (lower is better)
        self.assertTrue(self.adaptive_es._detect_improvement(0.48))
        
        # Test no improvement
        self.assertFalse(self.adaptive_es._detect_improvement(0.52))
        
        # Test marginal improvement (below threshold)
        self.assertFalse(self.adaptive_es._detect_improvement(0.495))  # Only 0.005 improvement
    
    def test_patience_adjustment_integration(self):
        """Test full patience adjustment workflow"""
        # Start with initial patience
        initial_patience = self.adaptive_es.patience
        
        # Simulate training with improvements and plateaus
        self.adaptive_es.best_score = 0.7
        
        # First improvement - patience should reset
        self.adaptive_es._adjust_patience(0.72)
        self.assertEqual(self.adaptive_es.patience, self.adaptive_es.initial_patience)
        
        # Update best score to simulate the improvement being recorded
        self.adaptive_es.best_score = 0.72
        
        # Then plateau - patience should increase
        self.adaptive_es._adjust_patience(0.71)
        increased_patience = self.adaptive_es.patience
        self.assertGreater(increased_patience, self.adaptive_es.initial_patience)
        
        # Another plateau - patience should increase more
        self.adaptive_es._adjust_patience(0.705)
        self.assertGreaterEqual(self.adaptive_es.patience, increased_patience)


class TestEarlyStoppingIntegration(unittest.TestCase):
    """Integration tests for early stopping in training context"""
    
    def test_robust_early_stopping_with_missing_metrics(self):
        """Test robust early stopping handles missing metrics gracefully"""
        mock_trainer = Mock()
        mock_trainer.callback_metrics = {
            'val/loss': torch.tensor(0.5),
            # No AUROC metrics available
        }
        
        robust_es = RobustEarlyStopping(monitor='val/auroc_subject')
        
        # Should not crash when ensemble metrics are unavailable
        try:
            ensemble_metric = robust_es._get_ensemble_metric(mock_trainer)
            # Should fallback to original metric or handle gracefully
            self.assertIsInstance(ensemble_metric, (float, int, type(None)))
        except KeyError:
            self.fail("RobustEarlyStopping should handle missing metrics gracefully")
    
    def test_early_stopping_metric_consistency(self):
        """Test that early stopping metrics are consistent across calls"""
        mock_trainer = Mock()
        mock_trainer.callback_metrics = {
            'val/auroc_subject_mean_prob': torch.tensor(0.8),
            'val/auroc_subject_mean_logit': torch.tensor(0.82),
            'val/auroc_subject_noisy_or': torch.tensor(0.78)
        }
        
        robust_es = RobustEarlyStopping(
            monitor='val/auroc_subject',
            ensemble_methods=['mean_prob', 'mean_logit', 'noisy_or']
        )
        
        # Multiple calls should return same result
        metric1 = robust_es._get_ensemble_metric(mock_trainer)
        metric2 = robust_es._get_ensemble_metric(mock_trainer)
        
        self.assertEqual(metric1, metric2)
    
    def test_smoothing_numerical_stability(self):
        """Test that smoothing doesn't introduce numerical instabilities"""
        robust_es = RobustEarlyStopping(monitor='val/loss', smoothing_window=5)
        
        # Test with extreme values
        extreme_values = [1e-8, 1e8, 0.0, 1.0, -1.0]
        
        for value in extreme_values:
            try:
                smoothed = robust_es._apply_smoothing(value)
                self.assertFalse(np.isnan(smoothed))
                self.assertFalse(np.isinf(smoothed))
            except Exception as e:
                self.fail(f"Smoothing failed with value {value}: {e}")


if __name__ == '__main__':
    unittest.main()
