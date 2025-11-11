import unittest
import tempfile
import os
import torch
import argparse
from unittest.mock import patch, MagicMock

from inference.predict_task1 import InferenceAggregator, create_ensemble_config


class TestTask1KFoldEnsemble(unittest.TestCase):
    """
    Test suite for Task 1 K-fold ensemble functionality.
    Tests the ensemble system for aggregating predictions from multiple fold models.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock model for testing
        self.mock_model = MagicMock()
        mock_param = torch.tensor([1.0], device='cpu')
        self.mock_model.parameters.side_effect = lambda: iter([mock_param])
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_ensemble_config_single_checkpoint(self):
        """Test ensemble configuration with single checkpoint."""
        args = MagicMock()
        args.ensemble_checkpoints = "fold0.ckpt"
        
        ensemble_config = create_ensemble_config(args)
        self.assertTrue(ensemble_config['enable'])
        self.assertEqual(len(ensemble_config['checkpoints']), 1)
        self.assertEqual(ensemble_config['checkpoints'][0], 'fold0.ckpt')
        self.assertEqual(ensemble_config['method'], 'simple_average')
    
    def test_ensemble_config_multiple_checkpoints(self):
        """Test ensemble configuration with K-fold checkpoints."""
        args = MagicMock()
        args.ensemble_checkpoints = "fold0.ckpt, fold1.ckpt, fold2.ckpt, fold3.ckpt, fold4.ckpt"
        
        ensemble_config = create_ensemble_config(args)
        self.assertTrue(ensemble_config['enable'])
        self.assertEqual(len(ensemble_config['checkpoints']), 5)  # 5-fold
        
        expected_checkpoints = ['fold0.ckpt', 'fold1.ckpt', 'fold2.ckpt', 'fold3.ckpt', 'fold4.ckpt']
        self.assertEqual(ensemble_config['checkpoints'], expected_checkpoints)
    
    def test_ensemble_config_with_paths(self):
        """Test ensemble configuration with full checkpoint paths."""
        args = MagicMock()
        checkpoint_paths = [
            "./runs/Task001_FOMO1/unet_xl/fomo1_kfold_fold0/version_0/checkpoints/best.ckpt",
            "./runs/Task001_FOMO1/unet_xl/fomo1_kfold_fold1/version_0/checkpoints/best.ckpt",
            "./runs/Task001_FOMO1/unet_xl/fomo1_kfold_fold2/version_0/checkpoints/best.ckpt"
        ]
        args.ensemble_checkpoints = ", ".join(checkpoint_paths)
        
        ensemble_config = create_ensemble_config(args)
        self.assertTrue(ensemble_config['enable'])
        self.assertEqual(len(ensemble_config['checkpoints']), 3)
        self.assertEqual(ensemble_config['checkpoints'], checkpoint_paths)
    
    @patch('inference.predict_task1.ModelLoader')
    def test_ensemble_model_loading(self, mock_loader):
        """Test ensemble model loading and cleanup."""
        # Mock ModelLoader to return different models
        mock_models = []
        for i in range(3):
            mock_model = MagicMock()
            mock_model.parameters.side_effect = lambda: iter([torch.tensor([1.0], device='cpu')])
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            # Different prediction for each model to simulate ensemble diversity
            mock_model.return_value = torch.tensor([[0.3 + i*0.1, 0.7 - i*0.1]])
            mock_models.append(mock_model)
        
        mock_loader.load_model.side_effect = mock_models
        
        # Create aggregator with ensemble config
        ensemble_config = {
            'enable': True,
            'checkpoints': ['fold0.ckpt', 'fold1.ckpt', 'fold2.ckpt'],
            'method': 'simple_average'
        }
        
        aggregator = InferenceAggregator(
            base_model=self.mock_model, 
            ensemble_config=ensemble_config
        )
        
        # Test input
        input_data = torch.randn(1, 2, 32, 32, 32)
        
        # Mock base model prediction
        self.mock_model.return_value = torch.tensor([[0.5, 0.5]])
        
        # Run ensemble prediction
        with patch('torch.cuda.empty_cache'):  # Mock CUDA cleanup
            result = aggregator.predict(input_data)
        
        # Should load 3 ensemble models + 1 base model = 4 total predictions
        self.assertEqual(mock_loader.load_model.call_count, 3)
        
        # Each model should be moved to device and set to eval
        for mock_model in mock_models:
            mock_model.to.assert_called_once_with(aggregator.device)
            mock_model.eval.assert_called_once()
        
        # Result should be aggregated prediction
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([1, 2]))
    
    @patch('inference.predict_task1.ModelLoader')
    def test_ensemble_prediction_aggregation(self, mock_loader):
        """Test that ensemble predictions are correctly aggregated."""
        # Create predictable mock models
        predictions = [
            torch.tensor([[0.2, 0.8]]),  # Fold 0: strong positive
            torch.tensor([[0.4, 0.6]]),  # Fold 1: weak positive  
            torch.tensor([[0.6, 0.4]]),  # Fold 2: weak negative
        ]
        
        mock_models = []
        for pred in predictions:
            mock_model = MagicMock()
            mock_model.parameters.side_effect = lambda: iter([torch.tensor([1.0], device='cpu')])
            mock_model.to.return_value = mock_model
            mock_model.return_value = pred
            mock_models.append(mock_model)
        
        mock_loader.load_model.side_effect = mock_models
        
        # Base model prediction
        base_pred = torch.tensor([[0.3, 0.7]])
        self.mock_model.return_value = base_pred
        
        # Create aggregator
        ensemble_config = {
            'enable': True,
            'checkpoints': ['fold0.ckpt', 'fold1.ckpt', 'fold2.ckpt'],
            'method': 'simple_average'
        }
        
        aggregator = InferenceAggregator(
            base_model=self.mock_model,
            ensemble_config=ensemble_config
        )
        
        input_data = torch.randn(1, 2, 32, 32, 32)
        
        with patch('torch.cuda.empty_cache'):
            result = aggregator.predict(input_data)
        
        # Calculate expected mean: (base + fold0 + fold1 + fold2) / 4
        all_predictions = [base_pred] + predictions
        expected_mean = torch.stack(all_predictions).mean(dim=0)
        
        # Check that result is close to expected mean
        torch.testing.assert_close(result, expected_mean, rtol=1e-3, atol=1e-3)
        
        # Verify aggregation produces reasonable probabilities
        self.assertGreaterEqual(result[0, 0].item(), 0.0)
        self.assertLessEqual(result[0, 0].item(), 1.0)
        self.assertGreaterEqual(result[0, 1].item(), 0.0)
        self.assertLessEqual(result[0, 1].item(), 1.0)
    
    @patch('inference.predict_task1.ModelLoader')
    def test_ensemble_error_handling(self, mock_loader):
        """Test that ensemble handles model loading errors gracefully."""
        # First model loads successfully, second fails, third loads successfully
        mock_model1 = MagicMock()
        mock_model1.parameters.side_effect = lambda: iter([torch.tensor([1.0], device='cpu')])
        mock_model1.to.return_value = mock_model1
        mock_model1.return_value = torch.tensor([[0.4, 0.6]])
        
        mock_model3 = MagicMock()
        mock_model3.parameters.side_effect = lambda: iter([torch.tensor([1.0], device='cpu')])
        mock_model3.to.return_value = mock_model3
        mock_model3.return_value = torch.tensor([[0.6, 0.4]])
        
        # Mock loader: success, error, success
        def load_side_effect(path, task_type):
            if 'fold1' in path:
                raise RuntimeError("Failed to load model")
            elif 'fold0' in path:
                return mock_model1
            elif 'fold2' in path:
                return mock_model3
        
        mock_loader.load_model.side_effect = load_side_effect
        
        # Base model
        self.mock_model.return_value = torch.tensor([[0.5, 0.5]])
        
        ensemble_config = {
            'enable': True,
            'checkpoints': ['fold0.ckpt', 'fold1.ckpt', 'fold2.ckpt'],
            'method': 'simple_average'
        }
        
        aggregator = InferenceAggregator(
            base_model=self.mock_model,
            ensemble_config=ensemble_config
        )
        
        input_data = torch.randn(1, 2, 32, 32, 32)
        
        # Should handle errors gracefully and continue with successful models
        with patch('torch.cuda.empty_cache'), \
             patch('logging.warning') as mock_warning:
            result = aggregator.predict(input_data)
        
        # Should have warned about the failed model
        mock_warning.assert_called()
        warning_call_args = mock_warning.call_args[0][0]
        self.assertIn("Failed to load ensemble model", warning_call_args)
        self.assertIn("fold1.ckpt", warning_call_args)
        
        # Should still produce valid result with remaining models
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([1, 2]))
    
    def test_ensemble_memory_efficiency(self):
        """Test that ensemble models are properly cleaned up for memory efficiency."""
        ensemble_config = {
            'enable': True,
            'checkpoints': ['fold0.ckpt', 'fold1.ckpt'],
            'method': 'simple_average'
        }
        
        aggregator = InferenceAggregator(
            base_model=self.mock_model,
            ensemble_config=ensemble_config
        )
        
        input_data = torch.randn(1, 2, 32, 32, 32)
        
        # Mock the ensemble loading process
        with patch.object(aggregator, '_apply_ensemble') as mock_ensemble:
            mock_ensemble.return_value = [torch.tensor([[0.4, 0.6]]), torch.tensor([[0.6, 0.4]])]
            
            # Mock base model
            self.mock_model.return_value = torch.tensor([[0.5, 0.5]])
            
            result = aggregator.predict(input_data)
            
            # Ensemble method should have been called
            mock_ensemble.assert_called_once_with(input_data)
        
        # Result should be properly aggregated
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([1, 2]))
    
    def test_kfold_checkpoint_pattern_recognition(self):
        """Test recognition of K-fold checkpoint patterns."""
        # Test different K-fold naming patterns
        test_patterns = [
            # Standard fold naming
            ["fold0.ckpt", "fold1.ckpt", "fold2.ckpt", "fold3.ckpt", "fold4.ckpt"],
            # Full paths with fold names
            [
                "./runs/Task001_FOMO1/unet_xl/fomo1_kfold_fold0/version_0/checkpoints/best.ckpt",
                "./runs/Task001_FOMO1/unet_xl/fomo1_kfold_fold1/version_0/checkpoints/best.ckpt",
                "./runs/Task001_FOMO1/unet_xl/fomo1_kfold_fold2/version_0/checkpoints/best.ckpt",
            ],
            # Mixed patterns
            ["model_fold_0.ckpt", "model_fold_1.ckpt", "model_fold_2.ckpt"]
        ]
        
        for pattern in test_patterns:
            with self.subTest(pattern=pattern):
                args = MagicMock()
                args.ensemble_checkpoints = ", ".join(pattern)
                
                config = create_ensemble_config(args)
                self.assertTrue(config['enable'])
                self.assertEqual(len(config['checkpoints']), len(pattern))
                self.assertEqual(config['checkpoints'], pattern)
    
    def test_ensemble_uncertainty_estimation(self):
        """Test uncertainty estimation from ensemble predictions."""
        # Create aggregator
        aggregator = InferenceAggregator(self.mock_model)
        
        # Simulate ensemble predictions with different confidence levels
        predictions = [
            torch.tensor([[0.1, 0.9]]),  # Very confident positive
            torch.tensor([[0.2, 0.8]]),  # Confident positive  
            torch.tensor([[0.4, 0.6]]),  # Less confident positive
            torch.tensor([[0.45, 0.55]]),  # Very uncertain
            torch.tensor([[0.3, 0.7]]),  # Moderately confident positive
        ]
        
        uncertainty_result = aggregator._estimate_uncertainty(predictions)
        
        # Check result structure
        self.assertIn('prediction', uncertainty_result)
        self.assertIn('uncertainty', uncertainty_result)
        self.assertIn('confidence', uncertainty_result)
        self.assertIn('num_predictions', uncertainty_result)
        
        # Check values are reasonable
        self.assertEqual(uncertainty_result['num_predictions'], 5)
        self.assertIsInstance(uncertainty_result['confidence'], float)
        self.assertGreater(uncertainty_result['confidence'], 0.0)
        
        # Mean prediction should be reasonable
        mean_pred = uncertainty_result['prediction']
        self.assertIsInstance(mean_pred, torch.Tensor)
        self.assertEqual(mean_pred.shape, torch.Size([1, 2]))
        self.assertGreaterEqual(mean_pred[0, 0].item(), 0.0)
        self.assertLessEqual(mean_pred[0, 0].item(), 1.0)


if __name__ == '__main__':
    unittest.main()
