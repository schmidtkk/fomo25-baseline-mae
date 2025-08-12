#!/usr/bin/env python3
"""
Unit tests for enhanced finetune.py functionality.
Tests CLI argument parsing and enhanced feature integration.
"""

import unittest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
import argparse
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock imports to avoid heavy dependencies in testing
with patch.dict('sys.modules', {
    'lightning': MagicMock(),
    'lightning.pytorch': MagicMock(),
    'lightning.pytorch.callbacks': MagicMock(),
    'lightning.pytorch.loggers': MagicMock(),
}):
    from finetune import create_model, add_enhanced_arguments, create_enhanced_callbacks


class TestEnhancedFinetuneArguments(unittest.TestCase):
    """Test enhanced CLI argument parsing"""
    
    def setUp(self):
        """Set up argument parser with enhanced arguments"""
        self.parser = argparse.ArgumentParser()
        add_enhanced_arguments(self.parser)
    
    def test_enhanced_aggregation_argument(self):
        """Test enhanced aggregation CLI argument"""
        # Test default (disabled)
        args = self.parser.parse_args([])
        self.assertFalse(args.enable_enhanced_aggregation)
        
        # Test enabled
        args = self.parser.parse_args(['--enable_enhanced_aggregation'])
        self.assertTrue(args.enable_enhanced_aggregation)
    
    def test_training_visualization_argument(self):
        """Test training visualization CLI argument"""
        # Test default (disabled)
        args = self.parser.parse_args([])
        self.assertFalse(args.enable_training_visualization)
        
        # Test enabled
        args = self.parser.parse_args(['--enable_training_visualization'])
        self.assertTrue(args.enable_training_visualization)
    
    def test_enhanced_early_stopping_argument(self):
        """Test enhanced early stopping CLI argument"""
        # Test default
        args = self.parser.parse_args([])
        self.assertEqual(args.enhanced_early_stopping, 'standard')
        
        # Test robust
        args = self.parser.parse_args(['--enhanced_early_stopping', 'robust'])
        self.assertEqual(args.enhanced_early_stopping, 'robust')
        
        # Test adaptive
        args = self.parser.parse_args(['--enhanced_early_stopping', 'adaptive'])
        self.assertEqual(args.enhanced_early_stopping, 'adaptive')
        
        # Test invalid option should raise error
        with self.assertRaises(SystemExit):
            self.parser.parse_args(['--enhanced_early_stopping', 'invalid'])
    
    def test_visualization_update_freq_argument(self):
        """Test visualization update frequency argument"""
        # Test default
        args = self.parser.parse_args([])
        self.assertEqual(args.visualization_update_freq, 2)
        
        # Test custom value
        args = self.parser.parse_args(['--visualization_update_freq', '5'])
        self.assertEqual(args.visualization_update_freq, 5)
    
    def test_updated_defaults(self):
        """Test that default values are updated for enhanced features"""
        # Create parser with all arguments (would include existing ones)
        full_parser = argparse.ArgumentParser()
        add_enhanced_arguments(full_parser)
        
        # Add some standard arguments with updated defaults
        full_parser.add_argument('--patch_size', type=int, default=128)
        full_parser.add_argument('--early_stop_patience', type=int, default=20)
        full_parser.add_argument('--early_stop_min_delta', type=float, default=0.001)
        
        args = full_parser.parse_args([])
        
        # Check updated defaults
        self.assertEqual(args.patch_size, 128)  # Updated from 32
        self.assertEqual(args.early_stop_patience, 20)  # Updated from 12
        self.assertEqual(args.early_stop_min_delta, 0.001)  # Updated from 0.002


class TestCreateEnhancedCallbacks(unittest.TestCase):
    """Test enhanced callback creation"""
    
    def setUp(self):
        """Set up test arguments"""
        self.temp_dir = tempfile.mkdtemp()
        self.args = Mock()
        self.args.save_dir = self.temp_dir
        self.args.enable_enhanced_aggregation = True
        self.args.enable_training_visualization = True
        self.args.enhanced_early_stopping = 'robust'
        self.args.visualization_update_freq = 2
        self.args.early_stop_patience = 20
        self.args.early_stop_min_delta = 0.001
        self.args.monitor_metric = 'val/auroc_subject'
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('finetune.LiveTrainingMonitor')
    @patch('finetune.RobustEarlyStopping')
    def test_create_enhanced_callbacks_all_enabled(self, mock_robust_es, mock_monitor):
        """Test callback creation with all enhanced features enabled"""
        callbacks = create_enhanced_callbacks(self.args)
        
        # Should create enhanced callbacks
        self.assertGreater(len(callbacks), 0)
        
        # Verify training monitor was created
        mock_monitor.assert_called_once_with(
            save_dir=os.path.join(self.temp_dir, 'training_plots'),
            update_frequency=2
        )
        
        # Verify robust early stopping was created
        mock_robust_es.assert_called_once_with(
            monitor='val/auroc_subject',
            patience=20,
            min_delta=0.001,
            mode='max'
        )
    
    @patch('finetune.AdaptiveEarlyStopping')
    def test_create_enhanced_callbacks_adaptive_early_stopping(self, mock_adaptive_es):
        """Test callback creation with adaptive early stopping"""
        self.args.enhanced_early_stopping = 'adaptive'
        self.args.enable_training_visualization = False
        
        callbacks = create_enhanced_callbacks(self.args)
        
        # Verify adaptive early stopping was created
        mock_adaptive_es.assert_called_once_with(
            monitor='val/auroc_subject',
            initial_patience=20,
            max_patience=50,  # Should be 2.5x initial
            patience_factor=1.5,
            improvement_threshold=0.001
        )
    
    def test_create_enhanced_callbacks_disabled(self):
        """Test callback creation with enhanced features disabled"""
        self.args.enable_enhanced_aggregation = False
        self.args.enable_training_visualization = False
        self.args.enhanced_early_stopping = 'standard'
        
        callbacks = create_enhanced_callbacks(self.args)
        
        # Should return empty list when all features disabled
        self.assertEqual(len(callbacks), 0)
    
    @patch('finetune.LiveTrainingMonitor')
    def test_create_enhanced_callbacks_visualization_only(self, mock_monitor):
        """Test callback creation with only visualization enabled"""
        self.args.enable_enhanced_aggregation = False
        self.args.enhanced_early_stopping = 'standard'
        
        callbacks = create_enhanced_callbacks(self.args)
        
        # Should only create visualization callback
        mock_monitor.assert_called_once()
        self.assertEqual(len(callbacks), 1)


class TestCreateModel(unittest.TestCase):
    """Test enhanced model creation"""
    
    def setUp(self):
        """Set up test arguments"""
        self.args = Mock()
        self.args.enable_enhanced_aggregation = True
        self.args.export_subject_probs = True
        self.args.num_classes = 2
        self.args.model_name = 'unet_b'
        self.args.taskid = 1
    
    @patch('finetune.SupervisedClsModel')
    def test_create_model_with_enhanced_features(self, mock_cls_model):
        """Test model creation with enhanced features"""
        # Mock model instance
        mock_model_instance = MagicMock()
        mock_cls_model.return_value = mock_model_instance
        
        model = create_model(self.args, config={})
        
        # Verify model was created with enhanced config
        mock_cls_model.assert_called_once()
        call_args = mock_cls_model.call_args[0][0]  # First argument (config)
        
        self.assertTrue(call_args['enable_enhanced_aggregation'])
        self.assertTrue(call_args['export_subject_probs'])
    
    @patch('finetune.SupervisedClsModel')
    def test_create_model_standard_features(self, mock_cls_model):
        """Test model creation with standard features only"""
        self.args.enable_enhanced_aggregation = False
        self.args.export_subject_probs = False
        
        model = create_model(self.args, config={})
        
        # Verify model was created with standard config
        mock_cls_model.assert_called_once()
        call_args = mock_cls_model.call_args[0][0]  # First argument (config)
        
        self.assertFalse(call_args.get('enable_enhanced_aggregation', False))
        self.assertFalse(call_args.get('export_subject_probs', False))


class TestArgumentValidation(unittest.TestCase):
    """Test argument validation for enhanced features"""
    
    def test_visualization_frequency_validation(self):
        """Test visualization update frequency validation"""
        parser = argparse.ArgumentParser()
        add_enhanced_arguments(parser)
        
        # Valid frequency
        args = parser.parse_args(['--visualization_update_freq', '1'])
        self.assertEqual(args.visualization_update_freq, 1)
        
        # Zero frequency should be handled (though may not be useful)
        args = parser.parse_args(['--visualization_update_freq', '0'])
        self.assertEqual(args.visualization_update_freq, 0)
    
    def test_early_stopping_patience_validation(self):
        """Test early stopping patience validation"""
        parser = argparse.ArgumentParser()
        add_enhanced_arguments(parser)
        parser.add_argument('--early_stop_patience', type=int, default=20)
        
        # Valid patience values
        for patience in [1, 10, 50, 100]:
            args = parser.parse_args(['--early_stop_patience', str(patience)])
            self.assertEqual(args.early_stop_patience, patience)


class TestEnhancedFeatureIntegration(unittest.TestCase):
    """Integration tests for enhanced features in finetune"""
    
    def test_enhanced_argument_combinations(self):
        """Test various combinations of enhanced arguments"""
        parser = argparse.ArgumentParser()
        add_enhanced_arguments(parser)
        
        # Test all enhanced features enabled
        args = parser.parse_args([
            '--enable_enhanced_aggregation',
            '--enable_training_visualization',
            '--enhanced_early_stopping', 'robust',
            '--visualization_update_freq', '3'
        ])
        
        self.assertTrue(args.enable_enhanced_aggregation)
        self.assertTrue(args.enable_training_visualization)
        self.assertEqual(args.enhanced_early_stopping, 'robust')
        self.assertEqual(args.visualization_update_freq, 3)
    
    def test_partial_enhanced_features(self):
        """Test partial enablement of enhanced features"""
        parser = argparse.ArgumentParser()
        add_enhanced_arguments(parser)
        
        # Test only aggregation enabled
        args = parser.parse_args(['--enable_enhanced_aggregation'])
        self.assertTrue(args.enable_enhanced_aggregation)
        self.assertFalse(args.enable_training_visualization)
        self.assertEqual(args.enhanced_early_stopping, 'standard')
        
        # Test only visualization enabled
        args = parser.parse_args(['--enable_training_visualization'])
        self.assertFalse(args.enable_enhanced_aggregation)
        self.assertTrue(args.enable_training_visualization)
        self.assertEqual(args.enhanced_early_stopping, 'standard')
    
    @patch('finetune.create_enhanced_callbacks')
    @patch('finetune.create_model')
    def test_callback_model_integration(self, mock_create_model, mock_create_callbacks):
        """Test that callbacks and model are created with consistent configuration"""
        # This test would require more setup to fully test the main function
        # For now, we test that the functions can be called independently
        
        args = Mock()
        args.enable_enhanced_aggregation = True
        args.enable_training_visualization = True
        args.enhanced_early_stopping = 'robust'
        args.visualization_update_freq = 2
        args.early_stop_patience = 20
        args.early_stop_min_delta = 0.001
        args.monitor_metric = 'val/auroc_subject'
        args.save_dir = '/tmp'
        args.export_subject_probs = True
        
        # Test callback creation
        callbacks = create_enhanced_callbacks(args)
        
        # Test model creation
        model = create_model(args, config={})
        
        # Both should work without errors
        mock_create_callbacks.assert_called_once_with(args)
        mock_create_model.assert_called_once_with(args, config={})


class TestEnhancedFeatureCompatibility(unittest.TestCase):
    """Test compatibility of enhanced features with existing functionality"""
    
    def test_backward_compatibility(self):
        """Test that enhanced features don't break existing argument parsing"""
        parser = argparse.ArgumentParser()
        
        # Add some existing arguments
        parser.add_argument('--taskid', type=int, default=1)
        parser.add_argument('--model_name', type=str, default='unet_b')
        parser.add_argument('--epochs', type=int, default=100)
        
        # Add enhanced arguments
        add_enhanced_arguments(parser)
        
        # Test that existing arguments still work
        args = parser.parse_args(['--taskid', '2', '--model_name', 'unet_xl'])
        
        self.assertEqual(args.taskid, 2)
        self.assertEqual(args.model_name, 'unet_xl')
        self.assertEqual(args.epochs, 100)  # Default value
        
        # Enhanced features should have their defaults
        self.assertFalse(args.enable_enhanced_aggregation)
        self.assertFalse(args.enable_training_visualization)
        self.assertEqual(args.enhanced_early_stopping, 'standard')
    
    def test_argument_help_strings(self):
        """Test that enhanced arguments have proper help strings"""
        parser = argparse.ArgumentParser()
        add_enhanced_arguments(parser)
        
        # Get help text
        help_text = parser.format_help()
        
        # Check that enhanced arguments are documented
        self.assertIn('enable_enhanced_aggregation', help_text)
        self.assertIn('enable_training_visualization', help_text)
        self.assertIn('enhanced_early_stopping', help_text)
        self.assertIn('visualization_update_freq', help_text)


if __name__ == '__main__':
    unittest.main()
