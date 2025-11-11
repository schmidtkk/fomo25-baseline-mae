"""
Unit tests for Task3 Inference Integration

Tests the inference pipeline for Task3 brain age regression,
including configuration loading, model loading, and prediction functionality.
"""

import torch
import tempfile
import os
import yaml
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


class TestTask3InferenceIntegration:
    """Test suite for Task3 inference integration."""

    @pytest.fixture
    def mock_config(self):
        """Mock Task3 configuration matching training setup."""
        return {
            'config': {
                'task_type': 'regression',
                'loss_type': 'mae',
                'age_normalization': True,
                'age_mean': 61.87,
                'age_std': 15.09,
                'patch_size': [256, 256, 32],
                'multi_encoder_modalities': ['T1', 'T2'],
                'fusion_type': 'attention',
                'modality_to_global_group': {
                    'T1': 't1',
                    'T2': 't2'
                },
                'val_tta_enable': True,
                'val_tta_views': 4,
                'val_tta_offsets': 3
            }
        }

    @pytest.fixture
    def mock_checkpoint_path(self):
        """Mock checkpoint path for testing."""
        return "runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/checkpoints/best.ckpt"

    def test_config_loading_from_hparams(self, mock_config):
        """Test loading of Task3 configuration from hparams.yaml."""
        # Mock the load_task3_config function
        with patch('inference.predict_task3.yaml.safe_load') as mock_yaml_load:
            with patch('builtins.open', mock_open(read_data="dummy")):
                mock_yaml_load.return_value = mock_config
                
                # Import here to avoid import errors during test discovery
                from inference.predict_task3 import load_task3_config
                
                loaded_config = load_task3_config("dummy_path.yaml")
                
                assert loaded_config['task_type'] == 'regression'
                assert loaded_config['age_normalization'] == True
                assert loaded_config['age_mean'] == 61.87
                assert loaded_config['age_std'] == 15.09
                assert loaded_config['loss_type'] == 'mae'

    def test_config_loading_error_handling(self):
        """Test error handling when config file doesn't exist or is malformed."""
        from inference.predict_task3 import load_task3_config
        
        # Test file not found
        with pytest.raises(FileNotFoundError):
            load_task3_config("/nonexistent/path/config.yaml")
        
        # Test malformed YAML
        with patch('builtins.open', mock_open(read_data="invalid: yaml: content: [")):
            with patch('inference.predict_task3.yaml.safe_load') as mock_yaml_load:
                mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")
                
                with pytest.raises(yaml.YAMLError):
                    load_task3_config("malformed.yaml")

    def test_age_denormalization_accuracy(self):
        """Test age denormalization functionality with Task3 parameters."""
        from inference.predict_task3 import denormalize_age
        
        # Test with actual Task3 normalization parameters
        age_mean = 61.87
        age_std = 15.09
        
        # Test various normalized ages
        test_cases = [
            (0.0, 61.87),      # Mean normalized age -> actual mean
            (1.0, 76.96),      # One std above mean
            (-1.0, 46.78),     # One std below mean
            (0.5, 69.415),     # Half std above mean
            (-0.5, 54.325),    # Half std below mean
        ]
        
        for normalized, expected in test_cases:
            denorm_age = denormalize_age(normalized, age_mean, age_std)
            assert abs(denorm_age - expected) < 0.01, (
                f"Denormalization failed: {normalized} -> {denorm_age}, expected {expected}"
            )

    def test_age_denormalization_edge_cases(self):
        """Test edge cases for age denormalization."""
        from inference.predict_task3 import denormalize_age
        
        age_mean = 61.87
        age_std = 15.09
        
        # Test extreme values
        very_young = denormalize_age(-3.0, age_mean, age_std)  # ~16 years
        very_old = denormalize_age(3.0, age_mean, age_std)     # ~107 years
        
        assert 10 <= very_young <= 25, f"Very young age seems unrealistic: {very_young}"
        assert 95 <= very_old <= 115, f"Very old age seems unrealistic: {very_old}"

    @patch('inference.predict.ModelLoader.load_model')
    @patch('inference.predict.load_modalities')
    @patch('inference.predict_task3.preprocess_modalities')
    def test_brain_age_prediction_pipeline(self, mock_preprocess, mock_load_modalities, mock_load_model):
        """Test complete brain age prediction pipeline with mocked components."""
        # Mock model that returns normalized age prediction
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([0.5])  # Normalized age
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_load_model.return_value = mock_model
        
        # Mock modality loading (T1 and T2)
        mock_nifti_img = MagicMock()
        mock_nifti_img.affine = torch.eye(4).numpy()
        # Add shape attribute that Yucca preprocessing expects (3D for medical images)
        mock_nifti_img.shape = (256, 256, 32)  # [H, W, D] format
        mock_load_modalities.return_value = [mock_nifti_img, mock_nifti_img]
        
        # Mock preprocessing
        mock_preprocess.return_value = torch.randn(1, 2, 32, 32, 32)  # [B, C, D, H, W]
        
        from inference.predict_task3 import predict_brain_age
        
        config = {
            'age_normalization': True,
            'age_mean': 61.87,
            'age_std': 15.09,
            'task_type': 'regression',
            'patch_size': [256, 256, 32],
            'val_tta_enable': False
        }
        
        # Test prediction
        result = predict_brain_age(
            t1_path="mock_t1.nii.gz",
            t2_path="mock_t2.nii.gz", 
            checkpoint_path="mock.ckpt",
            config=config
        )
        
        # Should return denormalized age in years
        expected_age = 0.5 * 15.09 + 61.87  # ~69.4 years
        assert abs(result - expected_age) < 0.1, (
            f"Predicted age {result} doesn't match expected {expected_age}"
        )
        
        # Verify function calls
        mock_load_model.assert_called_once()
        mock_load_modalities.assert_called_once_with(["mock_t1.nii.gz", "mock_t2.nii.gz"])
        mock_preprocess.assert_called_once()

    @patch('inference.predict_task3.load_task3_config')
    @patch('inference.predict_task3.predict_brain_age')
    def test_cli_interface_basic_usage(self, mock_predict, mock_load_config):
        """Test CLI interface with basic arguments."""
        mock_load_config.return_value = {'age_normalization': True}
        mock_predict.return_value = 69.4
        
        # Mock command line arguments
        test_args = [
            'predict_task3.py',
            '--modalities', '/path/to/t1.nii.gz', '/path/to/t2.nii.gz',
            '--output_dir', '/tmp/output'
        ]
        
        with patch('sys.argv', test_args):
            with patch('inference.predict_task3.argparse.ArgumentParser') as mock_parser:
                mock_args = MagicMock()
                mock_args.modalities = ['/path/to/t1.nii.gz', '/path/to/t2.nii.gz']
                mock_args.output_dir = '/tmp/output'
                mock_args.checkpoint_path = None
                mock_args.config_path = None
                mock_args.device = 'auto'
                mock_args.tta = False
                mock_args.batch_size = 1
                
                mock_parser.return_value.parse_args.return_value = mock_args
                
                # Import and test main function
                from inference.predict_task3 import main
                
                with patch('os.makedirs'):
                    with patch('builtins.open', mock_open()):
                        result = main()
                        
                        assert result == 0  # Success exit code
                        mock_predict.assert_called_once()

    def test_cli_validation_missing_modalities(self):
        """Test CLI validation when modalities are missing.""" 
        test_args = [
            'predict_task3.py',
            '--modalities', '/path/to/t1.nii.gz',  # Missing T2
            '--output_dir', '/tmp/output'
        ]
        
        with patch('sys.argv', test_args):
            from inference.predict_task3 import validate_args
            
            mock_args = MagicMock()
            mock_args.modalities = ['/path/to/t1.nii.gz']  # Only one modality
            
            with pytest.raises(ValueError, match="exactly 2 modalities"):
                validate_args(mock_args)

    def test_cli_validation_nonexistent_files(self):
        """Test CLI validation when input files don't exist."""
        from inference.predict_task3 import validate_args
        
        mock_args = MagicMock()
        mock_args.modalities = ['/nonexistent/t1.nii.gz', '/nonexistent/t2.nii.gz']
        mock_args.checkpoint_path = None
        mock_args.config_path = None
        
        with pytest.raises(FileNotFoundError):
            validate_args(mock_args)

    @patch('torch.cuda.is_available')
    def test_device_selection_logic(self, mock_cuda_available):
        """Test automatic device selection logic."""
        from inference.predict_task3 import select_device
        
        # Test CUDA available
        mock_cuda_available.return_value = True
        device = select_device('auto')
        assert 'cuda' in str(device)
        
        # Test CUDA not available
        mock_cuda_available.return_value = False
        device = select_device('auto')
        assert str(device) == 'cpu'
        
        # Test explicit device specification
        device = select_device('cpu')
        assert str(device) == 'cpu'

    @patch('inference.predict_task3.predict_brain_age')
    def test_tta_integration(self, mock_predict):
        """Test TTA (Test-Time Augmentation) integration."""
        mock_predict.return_value = 70.5
        
        from inference.predict_task3 import run_inference_with_tta
        
        config = {
            'val_tta_enable': True,
            'val_tta_views': 4,
            'val_tta_offsets': 3,
            'age_normalization': True,
            'age_mean': 61.87,
            'age_std': 15.09
        }
        
        # Test that TTA configuration is passed correctly
        result = run_inference_with_tta(
            t1_path="t1.nii.gz",
            t2_path="t2.nii.gz", 
            checkpoint_path="best.ckpt",
            config=config
        )
        
        assert isinstance(result, float)
        mock_predict.assert_called_once()
        
        # Verify TTA was enabled in the call
        call_config = mock_predict.call_args[1]['config']
        assert call_config['val_tta_enable'] == True

    def test_output_file_creation(self):
        """Test that output files are created correctly."""
        from inference.predict_task3 import save_prediction_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "prediction.txt")
            predicted_age = 69.42
            
            save_prediction_result(predicted_age, output_path)
            
            # Verify file was created
            assert os.path.exists(output_path)
            
            # Verify content is correct
            with open(output_path, 'r') as f:
                content = f.read().strip()
                assert content == "69.42"

    def test_batch_processing_capability(self):
        """Test capability to process multiple subjects in batch."""
        from inference.predict_task3 import predict_batch
        
        subjects = [
            {"t1": "subj1_t1.nii.gz", "t2": "subj1_t2.nii.gz", "id": "subj001"},
            {"t1": "subj2_t1.nii.gz", "t2": "subj2_t2.nii.gz", "id": "subj002"}
        ]
        
        config = {'batch_size': 2, 'age_normalization': True}
        
        with patch('inference.predict_task3.predict_brain_age') as mock_predict:
            mock_predict.side_effect = [67.3, 72.8]  # Different ages for each subject
            
            results = predict_batch(subjects, "checkpoint.ckpt", config)
            
            assert len(results) == 2
            assert results[0]["id"] == "subj001"
            assert results[0]["predicted_age"] == 67.3
            assert results[1]["id"] == "subj002"
            assert results[1]["predicted_age"] == 72.8

    @patch('inference.predict.ModelLoader.detect_model_architecture')
    def test_checkpoint_compatibility_validation(self, mock_detect):
        """Test validation of checkpoint compatibility with Task3."""
        mock_detect.return_value = {
            'architecture': 'unet_xl',
            'encoder_type': 'multi_encoder',
            'modalities': ['T1', 'T2'],
            'fusion_type': 'attention',
            'config': {'task_type': 'regression'}
        }
        
        from inference.predict_task3 import validate_checkpoint_compatibility
        
        # Should pass validation for correct Task3 checkpoint
        is_compatible = validate_checkpoint_compatibility("checkpoint.ckpt")
        assert is_compatible == True
        
        # Test incompatible checkpoint (wrong task type)
        mock_detect.return_value['config']['task_type'] = 'classification'
        is_compatible = validate_checkpoint_compatibility("checkpoint.ckpt")
        assert is_compatible == False

    def test_error_handling_corrupted_checkpoint(self):
        """Test error handling for corrupted or invalid checkpoints."""
        from inference.predict_task3 import load_model_safely
        
        # Test non-existent checkpoint
        with pytest.raises(FileNotFoundError):
            load_model_safely("/nonexistent/checkpoint.ckpt", {})
        
        # Test corrupted checkpoint (mock torch.load raising an exception)
        with patch('torch.load') as mock_load:
            mock_load.side_effect = RuntimeError("Checkpoint is corrupted")
            
            with pytest.raises(RuntimeError, match="Checkpoint is corrupted"):
                load_model_safely("corrupted.ckpt", {})


if __name__ == "__main__":
    pytest.main([__file__])
