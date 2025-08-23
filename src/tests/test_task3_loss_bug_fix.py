"""
Unit tests for Task3 Loss Bug Fix

Tests the tensor size mismatch fix for Task3 brain age regression training.
This test reproduces the bug before the fix and validates it's resolved after.
"""

import torch
import pytest
import warnings
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.supervised_reg import SupervisedRegModel
from data.task_configs import task3_config


class TestTask3LossBugFix:
    """Test suite for Task3 tensor size mismatch bug fix."""

    def test_tensor_shape_consistency_before_fix(self):
        """
        Test that reproduces the tensor size mismatch warning before fix.
        
        This test is designed to fail initially and pass after implementing
        the shape normalization fix in supervised_reg.py.
        """
        # Create mock config for Task3
        config = {
            **task3_config,
            "loss_type": "mae", 
            "age_normalization": True,
            "age_mean": 61.87,
            "age_std": 15.09,
            "val_tta_enable": False,  # Disable TTA for simpler testing
        }
        
        model = SupervisedRegModel(config=config)
        model.eval()
        
        # Mock batch with problematic shapes that trigger the warning
        batch_size = 2
        mock_batch = {
            "image": torch.randn(batch_size, 2, 32, 32, 32),  # [B, C, D, H, W]
            "label": torch.tensor([65.0, 72.0]),  # [B] - scalar ages
            "file_path": ["subject_001", "subject_002"]
        }
        
        # Capture warnings to detect the tensor size mismatch
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            with torch.no_grad():
                # This should trigger the tensor size mismatch warning before fix
                model.validation_step(mock_batch, 0)
            
            # Check if the specific warning was raised (before fix)
            size_mismatch_warnings = [
                warning for warning in w 
                if "target size" in str(warning.message) and "input size" in str(warning.message)
            ]
            
            # After implementing the fix, this assertion should pass (no warnings)
            assert len(size_mismatch_warnings) == 0, (
                f"Tensor size mismatch warning still present: {size_mismatch_warnings}"
            )

    def test_model_output_shape_standardization(self):
        """Test that model outputs correct shape for regression."""
        config = {
            **task3_config, 
            "loss_type": "mae",
            "use_multi_encoder": True,
            "multi_encoder_modalities": ["T1", "T2"],
            "fusion_type": "attention"
        }
        model = SupervisedRegModel(config=config)
        model.eval()
        
        # Test forward pass shape
        batch_size = 2
        inputs = torch.randn(batch_size, 2, 32, 32, 32)
        
        with torch.no_grad():
            output = model(inputs)
        
        # Output should be [B] for scalar regression (not [B, 1])
        expected_shape = torch.Size([batch_size])
        assert output.shape == expected_shape, (
            f"Expected output shape {expected_shape}, got {output.shape}. "
            f"Regression output should be squeezed to match target shape."
        )

    def test_loss_computation_compatibility(self):
        """Test that loss functions work with corrected tensor shapes."""
        config = {**task3_config}
        
        # Test all supported loss types
        for loss_type in ["mse", "mae", "huber"]:
            config["loss_type"] = loss_type
            model = SupervisedRegModel(config=config)
            
            # Test with matching tensor shapes
            output = torch.tensor([65.0, 72.0])  # [B] - predicted ages
            target = torch.tensor([63.0, 70.0])  # [B] - actual ages
            
            # Should compute loss without warnings or errors
            loss_fn_train, loss_fn_val = model._configure_losses()
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loss = loss_fn_val(output, target)
                
                # No size mismatch warnings should be generated
                size_warnings = [
                    warning for warning in w 
                    if "target size" in str(warning.message)
                ]
                assert len(size_warnings) == 0, (
                    f"Loss computation still generating warnings for {loss_type}: {size_warnings}"
                )
            
            assert torch.isfinite(loss), f"Loss should be finite for {loss_type}, got {loss}"
            assert loss.item() >= 0, f"Loss should be non-negative for {loss_type}"

    def test_validation_step_shape_handling(self):
        """Test validation step handles various input/output shape combinations."""
        config = {
            **task3_config,
            "loss_type": "mae",
            "val_tta_enable": False  # Test without TTA first
        }
        model = SupervisedRegModel(config=config)
        model.eval()
        
        # Test different batch sizes to ensure shape handling is robust
        for batch_size in [1, 2, 4]:
            mock_batch = {
                "image": torch.randn(batch_size, 2, 32, 32, 32),
                "label": torch.rand(batch_size) * 60 + 20,  # Ages 20-80
                "file_path": [f"subject_{i:03d}" for i in range(batch_size)]
            }
            
            with torch.no_grad():
                # Should execute without shape mismatch warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    
                    result = model.validation_step(mock_batch, 0)
                    
                    # Verify no tensor size warnings
                    size_warnings = [
                        warning for warning in w 
                        if "target size" in str(warning.message)
                    ]
                    assert len(size_warnings) == 0, (
                        f"Batch size {batch_size} generated shape warnings: {size_warnings}"
                    )

    def test_training_step_consistency(self):
        """Test that training step also handles shapes correctly."""
        config = {
            **task3_config,
            "loss_type": "mae",
            "age_normalization": True,
            "age_mean": 61.87,
            "age_std": 15.09
        }
        model = SupervisedRegModel(config=config)
        model.train()
        
        batch_size = 2
        mock_batch = {
            "image": torch.randn(batch_size, 2, 32, 32, 32),
            "label": torch.tensor([45.0, 78.0]),  # Different ages
            "file_path": ["subject_A", "subject_B"]
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Training step should also handle shapes correctly
            loss = model.training_step(mock_batch, 0)
            
            # No shape mismatch warnings in training
            size_warnings = [
                warning for warning in w 
                if "target size" in str(warning.message)
            ]
            assert len(size_warnings) == 0, (
                f"Training step generated shape warnings: {size_warnings}"
            )
            
            # Loss should be valid
            assert torch.isfinite(loss), "Training loss should be finite"

    def test_age_normalization_shape_consistency(self):
        """Test that age normalization doesn't introduce shape issues."""
        config = {
            **task3_config,
            "loss_type": "mae",
            "age_normalization": True,
            "age_mean": 61.87,
            "age_std": 15.09
        }
        model = SupervisedRegModel(config=config)
        
        # Test _process_batch method directly
        batch = {
            "image": torch.randn(2, 2, 32, 32, 32),
            "label": torch.tensor([45.0, 78.0]),  # Raw ages
            "file_path": ["subj1", "subj2"]
        }
        
        inputs, target, file_paths = model._process_batch(batch)
        
        # Target should maintain correct shape after normalization
        assert target.shape == torch.Size([2]), f"Normalized target has wrong shape: {target.shape}"
        assert target.dtype == torch.float32, "Target should be float32"
        
        # Values should be properly normalized
        expected_norm_target = (torch.tensor([45.0, 78.0]) - 61.87) / 15.09
        torch.testing.assert_close(target, expected_norm_target, rtol=1e-5, atol=1e-5)

    def test_tta_prediction_shape_consistency(self):
        """Test that TTA predictions maintain correct shapes."""
        config = {
            **task3_config,
            "loss_type": "mae",
            "val_tta_enable": True,
            "val_tta_views": 4,
            "val_tta_offsets": 3,
            "val_tta_batch_size": 2
        }
        model = SupervisedRegModel(config=config)
        model.eval()
        
        # Initialize TTA configuration
        model.on_validation_epoch_start()
        
        batch_size = 2
        inputs = torch.randn(batch_size, 2, 32, 32, 32)
        
        with torch.no_grad():
            # TTA prediction should maintain correct output shape
            tta_output = model._compute_tta_prediction(inputs)
            
            # Should have the same shape as regular prediction
            regular_output = model(inputs)
            
            assert tta_output.shape == regular_output.shape, (
                f"TTA output shape {tta_output.shape} doesn't match "
                f"regular output shape {regular_output.shape}"
            )
            
            # Both should be [B] for regression (scalar per sample)
            expected_shape = torch.Size([batch_size])
            assert tta_output.shape == expected_shape, (
                f"TTA output should have shape {expected_shape}, got {tta_output.shape}"
            )


if __name__ == "__main__":
    pytest.main([__file__])
