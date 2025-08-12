#!/usr/bin/env python3
"""
Unit tests for enhanced SupervisedClsModel functionality.
Tests enhanced aggregation methods integration and individual probability tracking.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.supervised_cls import SupervisedClsModel
from utils.subject_aggregation import SubjectAggregationSuite


class TestEnhancedSupervisedClsModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test model with enhanced features"""
        # Mock model config
        self.config = {
            'num_classes': 2,
            'enable_enhanced_aggregation': True,
            'export_subject_probs': True
        }
        
        # Mock network
        self.mock_network = MagicMock()
        self.mock_network.return_value = torch.randn(2, 2)  # [batch_size, num_classes]
        
        # Create model
        self.model = SupervisedClsModel(self.config)
        self.model.network = self.mock_network
        
        # Setup temporary directory for exports
        self.temp_dir = tempfile.mkdtemp()
        self.model.trainer = Mock()
        self.model.trainer.logger = Mock()
        self.model.trainer.logger.version = 0
        self.model.trainer.logger.log_dir = self.temp_dir
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validation_step_individual_tracking(self):
        """Test that validation_step tracks individual probabilities correctly"""
        # Mock batch data
        batch = {
            'image': torch.randn(2, 1, 32, 32, 32),
            'label': torch.tensor([1, 0]),
            'subject_id': ['subj_A', 'subj_B']
        }
        batch_idx = 0
        
        # Mock network output
        logits = torch.tensor([[0.1, 2.0], [-1.5, 0.3]])  # [batch_size, num_classes]
        self.mock_network.return_value = logits
        
        # Call validation_step
        result = self.model.validation_step(batch, batch_idx)
        
        # Check that individual probabilities are tracked
        self.assertTrue(hasattr(self.model, 'individual_probs'))
        self.assertTrue(hasattr(self.model, 'individual_targets'))
        self.assertTrue(hasattr(self.model, 'individual_subject_ids'))
        
        # Verify stored data
        self.assertEqual(len(self.model.individual_probs), 2)
        self.assertEqual(len(self.model.individual_targets), 2)
        self.assertEqual(len(self.model.individual_subject_ids), 2)
        
        # Check probability values (should be softmax of logits)
        expected_probs = torch.softmax(logits, dim=1)
        np.testing.assert_array_almost_equal(
            self.model.individual_probs[-2:],  # Last 2 entries
            expected_probs.cpu().numpy(),
            decimal=5
        )
    
    def test_on_validation_epoch_end_enhanced_aggregation(self):
        """Test enhanced aggregation computation in validation epoch end"""
        # Setup individual probability data
        self.model.individual_probs = [
            np.array([0.2, 0.8]),  # subj_A, crop 1
            np.array([0.1, 0.9]),  # subj_A, crop 2
            np.array([0.7, 0.3]),  # subj_B, crop 1
            np.array([0.8, 0.2]),  # subj_B, crop 2
            np.array([0.6, 0.4]),  # subj_B, crop 3
        ]
        
        self.model.individual_targets = [1, 1, 0, 0, 0]  # True labels
        self.model.individual_subject_ids = ['subj_A', 'subj_A', 'subj_B', 'subj_B', 'subj_B']
        
        # Mock the aggregation suite
        with patch('models.supervised_cls.SubjectAggregationSuite') as mock_suite_class:
            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            
            # Mock aggregation results
            mock_all_results = {
                'mean_prob': {'subj_A': 0.85, 'subj_B': 0.25},
                'mean_logit': {'subj_A': 0.88, 'subj_B': 0.22},
                'noisy_or': {'subj_A': 0.91, 'subj_B': 0.28}
            }
            mock_suite.aggregate_all_methods.return_value = mock_all_results
            
            # Mock AUROC computation
            mock_aurocs = {
                'mean_prob': 0.75,
                'mean_logit': 0.78,
                'noisy_or': 0.82
            }
            mock_suite.compute_method_aurocs.return_value = mock_aurocs
            
            # Call validation epoch end
            self.model.on_validation_epoch_end()
            
            # Verify aggregation suite was called correctly
            mock_suite.aggregate_all_methods.assert_called_once()
            mock_suite.compute_method_aurocs.assert_called_once()
            
            # Check that method-specific AUROCs were logged
            expected_log_calls = [
                ('val/auroc_subject_mean_prob', 0.75),
                ('val/auroc_subject_mean_logit', 0.78),
                ('val/auroc_subject_noisy_or', 0.82)
            ]
            
            # Verify logging calls (mock may aggregate calls)
            log_call_args = [call[0] for call in self.model.log.call_args_list]
            for expected_key, expected_value in expected_log_calls:
                # Find matching call
                matching_calls = [args for args in log_call_args if args[0] == expected_key]
                self.assertTrue(len(matching_calls) > 0, f"Expected log call for {expected_key} not found")
    
    def test_compute_enhanced_subject_aurocs(self):
        """Test enhanced subject AUROC computation method"""
        # Setup test data
        self.model.individual_probs = [
            np.array([0.2, 0.8]),  # subj_A
            np.array([0.1, 0.9]),  # subj_A  
            np.array([0.7, 0.3]),  # subj_B
            np.array([0.8, 0.2]),  # subj_B
        ]
        self.model.individual_targets = [1, 1, 0, 0]
        self.model.individual_subject_ids = ['subj_A', 'subj_A', 'subj_B', 'subj_B']
        
        # Call the method
        all_results, aurocs = self.model._compute_enhanced_subject_aurocs()
        
        # Check return types
        self.assertIsInstance(all_results, dict)
        self.assertIsInstance(aurocs, dict)
        
        # Check that we have results for multiple methods
        self.assertGreater(len(all_results), 1)
        self.assertGreater(len(aurocs), 1)
        
        # Check that all subjects are present in results
        for method_results in all_results.values():
            self.assertIn('subj_A', method_results)
            self.assertIn('subj_B', method_results)
        
        # Check AUROC values are reasonable
        for auroc in aurocs.values():
            self.assertTrue(0.0 <= auroc <= 1.0)
    
    def test_export_enhanced_subject_results(self):
        """Test enhanced subject results export"""
        # Setup test data
        all_results = {
            'mean_prob': {'subj_A': 0.85, 'subj_B': 0.25},
            'mean_logit': {'subj_A': 0.88, 'subj_B': 0.22},
            'noisy_or': {'subj_A': 0.91, 'subj_B': 0.28}
        }
        aurocs = {
            'mean_prob': 0.75,
            'mean_logit': 0.78,
            'noisy_or': 0.82
        }
        epoch = 5
        
        # Call export method
        self.model._export_enhanced_subject_results(all_results, aurocs, epoch)
        
        # Check that export file was created
        export_dir = os.path.join(self.temp_dir, 'subject_probs_enhanced')
        self.assertTrue(os.path.exists(export_dir))
        
        expected_file = os.path.join(export_dir, f'enhanced_results_epoch_{epoch:04d}.json')
        self.assertTrue(os.path.exists(expected_file))
        
        # Load and verify exported data
        with open(expected_file, 'r') as f:
            exported_data = json.load(f)
        
        self.assertEqual(exported_data['epoch'], epoch)
        self.assertIn('method_results', exported_data)
        self.assertIn('method_aurocs', exported_data)
        self.assertEqual(exported_data['method_results'], all_results)
        self.assertEqual(exported_data['method_aurocs'], aurocs)
    
    def test_group_by_subject(self):
        """Test grouping of individual predictions by subject"""
        # Setup test data with multiple crops per subject
        self.model.individual_probs = [
            np.array([0.2, 0.8]),  # subj_A, crop 1
            np.array([0.1, 0.9]),  # subj_A, crop 2  
            np.array([0.3, 0.7]),  # subj_A, crop 3
            np.array([0.7, 0.3]),  # subj_B, crop 1
            np.array([0.8, 0.2]),  # subj_B, crop 2
            np.array([0.9, 0.1]),  # subj_C, crop 1
        ]
        self.model.individual_targets = [1, 1, 1, 0, 0, 0]
        self.model.individual_subject_ids = ['subj_A', 'subj_A', 'subj_A', 'subj_B', 'subj_B', 'subj_C']
        
        # Get grouped data
        subject_probs, subject_targets = self.model._group_by_subject()
        
        # Check grouping
        self.assertEqual(len(subject_probs), 3)  # 3 unique subjects
        self.assertEqual(len(subject_targets), 3)
        
        # Check subject A has 3 crops
        self.assertEqual(len(subject_probs['subj_A']), 3)
        self.assertEqual(subject_targets['subj_A'], 1)
        
        # Check subject B has 2 crops  
        self.assertEqual(len(subject_probs['subj_B']), 2)
        self.assertEqual(subject_targets['subj_B'], 0)
        
        # Check subject C has 1 crop
        self.assertEqual(len(subject_probs['subj_C']), 1)
        self.assertEqual(subject_targets['subj_C'], 0)
        
        # Check probability values are correct (positive class probabilities)
        expected_subj_A_probs = [0.8, 0.9, 0.7]
        np.testing.assert_array_almost_equal(subject_probs['subj_A'], expected_subj_A_probs, decimal=5)
    
    def test_individual_data_reset(self):
        """Test that individual data is reset between epochs"""
        # Add some data
        self.model.individual_probs = [np.array([0.5, 0.5])]
        self.model.individual_targets = [1]
        self.model.individual_subject_ids = ['subj_A']
        
        # Call on_validation_epoch_start (should reset)
        self.model.on_validation_epoch_start()
        
        # Check that data was reset
        self.assertEqual(len(self.model.individual_probs), 0)
        self.assertEqual(len(self.model.individual_targets), 0)
        self.assertEqual(len(self.model.individual_subject_ids), 0)
    
    def test_backward_compatibility(self):
        """Test that enhanced features don't break standard functionality"""
        # Create model without enhanced features
        standard_config = {
            'num_classes': 2,
            'enable_enhanced_aggregation': False,
            'export_subject_probs': False
        }
        
        standard_model = SupervisedClsModel(standard_config)
        standard_model.network = self.mock_network
        
        # Test validation step works normally
        batch = {
            'image': torch.randn(2, 1, 32, 32, 32),
            'label': torch.tensor([1, 0]),
            'subject_id': ['subj_A', 'subj_B']
        }
        
        try:
            result = standard_model.validation_step(batch, 0)
            standard_model.on_validation_epoch_end()
        except Exception as e:
            self.fail(f"Standard functionality broken: {e}")
    
    def test_enhanced_features_enabled_check(self):
        """Test that enhanced features are only used when enabled"""
        # Test with enhanced features disabled
        self.model.config['enable_enhanced_aggregation'] = False
        
        # Setup some individual data (shouldn't be processed)
        self.model.individual_probs = [np.array([0.5, 0.5])]
        self.model.individual_targets = [1]
        self.model.individual_subject_ids = ['subj_A']
        
        # Mock the enhanced computation method to verify it's not called
        with patch.object(self.model, '_compute_enhanced_subject_aurocs') as mock_compute:
            self.model.on_validation_epoch_end()
            
            # Should not be called when disabled
            mock_compute.assert_not_called()
    
    def test_missing_subject_id_handling(self):
        """Test handling of missing subject IDs in batch"""
        # Batch without subject_id
        batch = {
            'image': torch.randn(2, 1, 32, 32, 32),
            'label': torch.tensor([1, 0])
            # Missing 'subject_id'
        }
        
        # Should handle gracefully or use fallback IDs
        try:
            result = self.model.validation_step(batch, 0)
        except Exception as e:
            self.fail(f"Missing subject_id handling failed: {e}")
    
    def test_empty_validation_data(self):
        """Test handling of empty validation data"""
        # No individual data collected
        self.assertEqual(len(self.model.individual_probs), 0)
        
        try:
            self.model.on_validation_epoch_end()
        except Exception as e:
            self.fail(f"Empty validation data handling failed: {e}")
    
    def test_single_subject_validation(self):
        """Test validation with only one subject (edge case)"""
        # Setup single subject data
        self.model.individual_probs = [
            np.array([0.2, 0.8]),
            np.array([0.1, 0.9])
        ]
        self.model.individual_targets = [1, 1]
        self.model.individual_subject_ids = ['subj_A', 'subj_A']
        
        try:
            all_results, aurocs = self.model._compute_enhanced_subject_aurocs()
            
            # AUROC computation should handle single class gracefully
            # (may be NaN or special value)
            for auroc in aurocs.values():
                self.assertTrue(np.isnan(auroc) or (0.0 <= auroc <= 1.0))
                
        except Exception as e:
            self.fail(f"Single subject validation failed: {e}")


class TestEnhancedModelIntegration(unittest.TestCase):
    """Integration tests for enhanced model features"""
    
    def test_full_validation_cycle(self):
        """Test full validation cycle with enhanced features"""
        config = {
            'num_classes': 2,
            'enable_enhanced_aggregation': True,
            'export_subject_probs': True
        }
        
        model = SupervisedClsModel(config)
        model.network = MagicMock()
        
        # Setup trainer
        model.trainer = Mock()
        model.trainer.logger = Mock()
        model.trainer.logger.version = 0
        
        temp_dir = tempfile.mkdtemp()
        model.trainer.logger.log_dir = temp_dir
        
        try:
            # Simulate validation epoch start
            model.on_validation_epoch_start()
            
            # Simulate multiple validation steps
            for i in range(5):
                batch = {
                    'image': torch.randn(2, 1, 32, 32, 32),
                    'label': torch.tensor([1, 0]),
                    'subject_id': [f'subj_{i}', f'subj_{i+5}']
                }
                
                # Mock network output
                model.network.return_value = torch.tensor([[0.1, 2.0], [-1.5, 0.3]])
                
                model.validation_step(batch, i)
            
            # Simulate validation epoch end
            model.on_validation_epoch_end()
            
            # Verify data was processed
            self.assertGreater(len(model.individual_probs), 0)
            
        except Exception as e:
            self.fail(f"Full validation cycle failed: {e}")
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
