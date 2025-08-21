import unittest
import torch

from models.supervised_reg import SupervisedRegModel


class TestRegressionTTA(unittest.TestCase):
    def test_tta_initialization(self):
        """Test TTA initialization for regression model"""
        cfg = {
            "num_classes": 1,
            "num_modalities": 2,
            "patch_size": (64, 64, 64),
            "model_name": "unet_xl",
            "version_dir": ".",
            "task_type": "regression",
            "val_tta_enable": True,
            "val_tta_views": 4,
            "val_tta_offsets": 3,
        }
        model = SupervisedRegModel(config=cfg, learning_rate=1e-4)
        model.on_validation_epoch_start()
        
        # Check TTA configuration
        self.assertTrue(model._val_tta_enable)
        self.assertEqual(len(model._tta_codes), 4)
        self.assertEqual(len(model._tta_offsets), 3)

    def test_tta_prediction_averaging(self):
        """Test TTA prediction averaging for regression"""
        cfg = {
            "num_classes": 1,
            "num_modalities": 2,
            "patch_size": (32, 32, 32),
            "model_name": "unet_xl",
            "version_dir": ".",
            "task_type": "regression",
            "val_tta_enable": True,
            "val_tta_views": 2,
            "val_tta_offsets": 1,
            "val_tta_batch_size": 2,
        }
        model = SupervisedRegModel(config=cfg, learning_rate=1e-4)
        model.on_validation_epoch_start()
        
        # Create test batch
        x = torch.randn(1, 2, 32, 32, 32)
        age = torch.tensor([65.5])  # Age in years
        batch = {"image": x, "label": age, "file_path": "FOMO3_sub_0001"}
        
        # Test validation step runs without error
        model.validation_step(batch, 0)
        
        # Test direct TTA computation
        tta_output = model._compute_tta_prediction(x)
        regular_output = model(x)
        
        # Outputs should have same shape
        self.assertEqual(tta_output.shape, regular_output.shape)
        
        # TTA output should be finite
        self.assertTrue(torch.isfinite(tta_output).all())

    def test_tta_memory_fallback(self):
        """Test TTA memory fallback mechanism"""
        cfg = {
            "num_classes": 1,
            "num_modalities": 2,
            "patch_size": (32, 32, 32),
            "model_name": "unet_xl",
            "version_dir": ".",
            "task_type": "regression",
            "val_tta_enable": True,
            "val_tta_views": 8,
            "val_tta_offsets": 7,
            "val_tta_batch_size": 1,  # Force small batches
        }
        model = SupervisedRegModel(config=cfg, learning_rate=1e-4)
        model.on_validation_epoch_start()
        
        x = torch.randn(1, 2, 32, 32, 32)
        
        # Should run without error even with many augmentations
        output = model._compute_tta_prediction(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_tta_disabled(self):
        """Test that TTA can be disabled"""
        cfg = {
            "num_classes": 1,
            "num_modalities": 2,
            "patch_size": (32, 32, 32),
            "model_name": "unet_xl",
            "version_dir": ".",
            "task_type": "regression",
            "val_tta_enable": False,
        }
        model = SupervisedRegModel(config=cfg, learning_rate=1e-4)
        model.on_validation_epoch_start()
        
        # TTA should be disabled
        self.assertFalse(model._val_tta_enable)
        
        x = torch.randn(1, 2, 32, 32, 32)
        age = torch.tensor([65.5])
        batch = {"image": x, "label": age, "file_path": "FOMO3_sub_0001"}
        
        # Should run validation without TTA
        model.validation_step(batch, 0)

    def test_age_normalization_with_tta(self):
        """Test TTA works with age normalization enabled"""
        cfg = {
            "num_classes": 1,
            "num_modalities": 2,
            "patch_size": (32, 32, 32),
            "model_name": "unet_xl",
            "version_dir": ".",
            "task_type": "regression",
            "val_tta_enable": True,
            "val_tta_views": 2,
            "age_normalization": True,
            "age_mean": 50.0,
            "age_std": 15.0,
        }
        model = SupervisedRegModel(config=cfg, learning_rate=1e-4)
        model.on_validation_epoch_start()
        
        x = torch.randn(1, 2, 32, 32, 32)
        age = torch.tensor([65.5])  # Raw age
        batch = {"image": x, "label": age, "file_path": "FOMO3_sub_0001"}
        
        # Should handle age normalization with TTA
        model.validation_step(batch, 0)


if __name__ == "__main__":
    unittest.main()
