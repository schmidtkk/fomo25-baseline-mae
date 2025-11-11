import unittest
import tempfile
import os
import torch
import argparse
from unittest.mock import patch, MagicMock

from inference.predict_task1 import InferenceAggregator, create_tta_config, create_ensemble_config


class TestTask1InferenceTTA(unittest.TestCase):
    """
    Test suite for Task 1 inference-time TTA functionality.
    Tests the inference TTA system for robust prediction aggregation.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock model for testing
        self.mock_model = MagicMock()
        # Fix parameters() to return a new iterator each time
        mock_param = torch.tensor([1.0], device='cpu')
        self.mock_model.parameters.side_effect = lambda: iter([mock_param])
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_tta_config_creation(self):
        """Test TTA configuration creation from command line arguments."""
        # Test TTA disabled
        args_no_tta = MagicMock()
        args_no_tta.tta_enable = False
        
        tta_config = create_tta_config(args_no_tta)
        self.assertFalse(tta_config['enable'])
        
        # Test TTA enabled with default settings
        args_tta = MagicMock()
        args_tta.tta_enable = True
        args_tta.tta_views = 8
        
        tta_config = create_tta_config(args_tta)
        self.assertTrue(tta_config['enable'])
        self.assertEqual(len(tta_config['transforms']), 8)
        self.assertEqual(tta_config['aggregation'], 'mean')
        
        # Check transform types
        expected_transforms = ['original', 'flip_x', 'flip_y', 'flip_z', 'flip_xy', 'flip_xz', 'flip_yz', 'flip_xyz']
        self.assertEqual(tta_config['transforms'], expected_transforms)
    
    def test_tta_config_reduced_views(self):
        """Test TTA configuration with reduced number of views."""
        args_tta = MagicMock()
        args_tta.tta_enable = True
        args_tta.tta_views = 4  # Reduced views
        
        tta_config = create_tta_config(args_tta)
        self.assertTrue(tta_config['enable'])
        self.assertEqual(len(tta_config['transforms']), 4)
        
        expected_transforms = ['original', 'flip_x', 'flip_y', 'flip_z']
        self.assertEqual(tta_config['transforms'], expected_transforms)
    
    def test_ensemble_config_creation(self):
        """Test ensemble configuration creation."""
        # Test ensemble disabled
        args_no_ensemble = MagicMock()
        args_no_ensemble.ensemble_checkpoints = None
        
        ensemble_config = create_ensemble_config(args_no_ensemble)
        self.assertFalse(ensemble_config['enable'])
        
        # Test ensemble enabled
        args_ensemble = MagicMock()
        args_ensemble.ensemble_checkpoints = "fold0.ckpt,fold1.ckpt,fold2.ckpt"
        
        ensemble_config = create_ensemble_config(args_ensemble)
        self.assertTrue(ensemble_config['enable'])
        self.assertEqual(len(ensemble_config['checkpoints']), 3)
        self.assertEqual(ensemble_config['checkpoints'], ['fold0.ckpt', 'fold1.ckpt', 'fold2.ckpt'])
        self.assertEqual(ensemble_config['method'], 'simple_average')
    
    def test_inference_aggregator_initialization(self):
        """Test InferenceAggregator initialization."""
        # Test basic initialization
        aggregator = InferenceAggregator(self.mock_model)
        self.assertEqual(aggregator.base_model, self.mock_model)
        self.assertEqual(aggregator.tta_config, {})
        self.assertEqual(aggregator.ensemble_config, {})
        
        # Test with TTA config
        tta_config = {
            'enable': True,
            'transforms': ['original', 'flip_x', 'flip_y'],
            'aggregation': 'mean'
        }
        
        aggregator = InferenceAggregator(self.mock_model, tta_config=tta_config)
        self.assertEqual(aggregator.tta_config, tta_config)
    
    def test_inference_aggregator_predict_no_tta(self):
        """Test inference prediction without TTA."""
        # Mock model output
        mock_output = torch.tensor([[0.3, 0.7]])  # Binary classification logits
        self.mock_model.return_value = mock_output
        
        # Create aggregator without TTA
        aggregator = InferenceAggregator(self.mock_model, tta_config={'enable': False})
        
        # Test input
        input_data = torch.randn(1, 2, 64, 64, 64)  # B, C=2 (DWI+FLAIR), D, H, W
        
        # Run prediction
        result = aggregator.predict(input_data)
        
        # Should call model once and return aggregated prediction
        self.mock_model.assert_called_once_with(input_data)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([1, 2]))  # Should maintain shape
    
    def test_inference_aggregator_tta_transforms(self):
        """Test TTA transform application."""
        # Mock model to return different outputs for different inputs
        def mock_model_side_effect(x):
            # Return different values based on input to simulate TTA effect
            return torch.tensor([[0.4, 0.6]]) + torch.randn(1, 2) * 0.1
        
        self.mock_model.side_effect = mock_model_side_effect
        
        # Create aggregator with TTA
        tta_config = {
            'enable': True,
            'transforms': ['original', 'flip_x', 'flip_y'],  # 3 transforms
            'aggregation': 'mean'
        }
        aggregator = InferenceAggregator(self.mock_model, tta_config=tta_config)
        
        # Test input
        input_data = torch.randn(1, 2, 32, 32, 32)
        
        # Run TTA prediction
        result = aggregator.predict(input_data)
        
        # Should call model multiple times (1 for base + 3 TTA transforms)
        self.assertGreaterEqual(self.mock_model.call_count, 3)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([1, 2]))
    
    def test_inference_tta_flip_transforms(self):
        """Test specific TTA flip transformations."""
        aggregator = InferenceAggregator(self.mock_model)
        
        # Test input with distinct pattern to verify flips
        input_data = torch.zeros(1, 2, 4, 4, 4)
        input_data[0, 0, 1, 1, 1] = 1.0  # Set a specific voxel
        
        # Test flip_x transform
        flipped_x = aggregator._apply_flip_transform(input_data, 'flip_x')
        self.assertEqual(flipped_x.shape, input_data.shape)
        # After flip_x, the voxel should move to a different x position
        self.assertNotEqual(flipped_x[0, 0, 1, 1, 1].item(), input_data[0, 0, 1, 1, 1].item())
        
        # Test flip_y transform  
        flipped_y = aggregator._apply_flip_transform(input_data, 'flip_y')
        self.assertEqual(flipped_y.shape, input_data.shape)
        
        # Test flip_z transform
        flipped_z = aggregator._apply_flip_transform(input_data, 'flip_z')
        self.assertEqual(flipped_z.shape, input_data.shape)
        
        # Test original (no transform)
        original = aggregator._apply_flip_transform(input_data, 'original')
        torch.testing.assert_close(original, input_data)
    
    def test_inference_aggregation_mean(self):
        """Test mean aggregation of TTA predictions."""
        aggregator = InferenceAggregator(self.mock_model)
        
        # Create mock predictions from different TTA views
        predictions = [
            torch.tensor([[0.2, 0.8]]),  # View 1
            torch.tensor([[0.4, 0.6]]),  # View 2  
            torch.tensor([[0.3, 0.7]]),  # View 3
        ]
        
        # Test mean aggregation
        result = aggregator._aggregate_predictions(predictions, method='mean')
        
        expected_mean = torch.mean(torch.stack(predictions), dim=0)
        torch.testing.assert_close(result, expected_mean)
        
        # Check that probabilities are reasonable
        self.assertGreaterEqual(result[0, 0].item(), 0.0)
        self.assertLessEqual(result[0, 0].item(), 1.0)
        self.assertGreaterEqual(result[0, 1].item(), 0.0) 
        self.assertLessEqual(result[0, 1].item(), 1.0)
    
    def test_inference_memory_efficiency(self):
        """Test that TTA inference handles memory efficiently."""
        # Mock model that simulates memory usage
        def mock_model_memory_test(x):
            # Simulate some computation
            return torch.randn(x.shape[0], 2)
        
        self.mock_model.side_effect = mock_model_memory_test
        
        tta_config = {
            'enable': True,
            'transforms': ['original', 'flip_x', 'flip_y', 'flip_z', 'flip_xy'],  # 5 transforms
            'aggregation': 'mean'
        }
        aggregator = InferenceAggregator(self.mock_model, tta_config=tta_config)
        
        # Test with reasonable input size
        input_data = torch.randn(1, 2, 32, 32, 32)
        
        # Should complete without memory errors
        try:
            result = aggregator.predict(input_data)
            success = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                success = False
            else:
                raise
        
        self.assertTrue(success)
        self.assertIsInstance(result, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
