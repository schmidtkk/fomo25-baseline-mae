import os
import sys
import tempfile
import unittest
import logging
import torch
import numpy as np
import json

from models.supervised_base import BaseSupervisedModel
from data.dataset_fusion import FusionCLSDataset


class TestFOMO2Segmentation(unittest.TestCase):
    """Test segmentation functionality for FOMO Task 2 (Meningioma Segmentation)"""
    
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.subject_dir = os.path.join(self.temp_dir, "test_subject")
        os.makedirs(self.subject_dir)
        
        # Create dummy modality data (32x32x32 volumes)
        self.shape = (32, 32, 32)
        self.modalities = ["DWI", "T2FLAIR", "SWI_OR_T2STAR"]
        
        for modality in self.modalities:
            data = np.random.randn(*self.shape).astype(np.float32)
            np.save(os.path.join(self.subject_dir, f"{modality}.npy"), data)
        
        # Create dummy segmentation mask (binary: background=0, meningioma=1)
        seg_mask = np.zeros(self.shape, dtype=np.int64)
        # Add a small meningioma region in the center
        seg_mask[14:18, 14:18, 14:18] = 1
        np.save(os.path.join(self.subject_dir, "seg.npy"), seg_mask)
        
        # Create mask.json
        mask_info = {
            "modalities": self.modalities,
            "mask": [1, 1, 1],  # All modalities present
            "present_modalities": self.modalities,
            "num_present": len(self.modalities)
        }
        with open(os.path.join(self.subject_dir, "mask.json"), "w") as f:
            json.dump(mask_info, f)

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _build_segmentation_config(self):
        """Build configuration for segmentation task"""
        return {
            "task": "Task002_FOMO2",
            "task_id": 2,
            "task_type": "segmentation",
            "experiment": "unittest_segmentation",
            "model_name": "unet_xl",
            "model_dimensions": "3D",
            "run_type": "finetune",
            "save_dir": self.temp_dir,
            "train_data_dir": self.temp_dir,
            "version_dir": self.temp_dir,
            "version": 0,
            "ckpt_path": None,
            "pretrained_weights_path": None,
            "seed": 0,
            "num_classes": 2,  # background + meningioma
            "num_modalities": 3,  # DWI, T2FLAIR, SWI_OR_T2STAR
            "image_extension": ".npy",
            "allow_missing_modalities": True,
            "labels": {0: "background", 1: "meningioma"},
            "batch_size": 1,
            "learning_rate": 1e-4,
            "patch_size": (32, 32, 32),
            "precision": "32-true",  # Use full precision for testing
            "augmentation_preset": "none",
            "epochs": 1,
            "train_batches_per_epoch": 1,
            "effective_batch_size": 1,
            "train_dataset_size": 1,
            "val_dataset_size": 1,
            "max_iterations": 1,
            "num_devices": 1,
            "num_workers": 0,
            "compile": False,
            "compile_mode": None,
            "fast_dev_run": True,
            "surface_metrics_enabled": False,  # Disable for unit test
            # Multi-encoder settings
            "use_multi_encoder": True,
            "multi_encoder_modalities": self.modalities,
            "modality_to_global_group": {
                "DWI": "dwi",
                "T2FLAIR": "flair",
                "SWI_OR_T2STAR": "other",
            },
            "global_vocab": ["t1", "t2", "flair", "dwi", "other"],
        }

    def test_segmentation_model_creation(self):
        """Test creation of segmentation model"""
        config = self._build_segmentation_config()
        model = BaseSupervisedModel.create(task_type="segmentation", config=config)
        
        # Check model type and configuration
        from models.supervised_seg import SupervisedSegModel
        self.assertIsInstance(model, SupervisedSegModel)
        self.assertEqual(model.task_type, "segmentation")
        self.assertEqual(model.num_classes, 2)

    def test_segmentation_dataset_loading(self):
        """Test segmentation dataset can load data properly"""
        dataset = FusionCLSDataset(
            samples=[self.subject_dir],
            patch_size=(32, 32, 32),
            task_type="segmentation"
        )
        
        # Test dataset length
        self.assertEqual(len(dataset), 1)
        
        # Test data loading
        data_dict = dataset[0]
        
        # Check image shape: [num_modalities, D, H, W]
        self.assertEqual(data_dict["image"].shape, (3, 32, 32, 32))
        
        # Check segmentation label shape: [D, H, W]
        self.assertEqual(data_dict["label"].shape, (32, 32, 32))
        
        # Check segmentation label type and values
        self.assertTrue(torch.is_tensor(data_dict["label"]))
        unique_values = torch.unique(data_dict["label"])
        self.assertTrue(all(v in [0, 1] for v in unique_values.tolist()))

    def test_segmentation_forward_pass(self):
        """Test forward pass through segmentation model"""
        config = self._build_segmentation_config()
        model = BaseSupervisedModel.create(task_type="segmentation", config=config)
        model.eval()
        
        # Create dummy batch
        batch_size = 2
        image = torch.randn(batch_size, 3, 32, 32, 32)  # [B, M, D, H, W]
        label = torch.randint(0, 2, (batch_size, 32, 32, 32))  # [B, D, H, W]
        
        batch = {
            "image": image,
            "label": label,
            "file_path": ["test1", "test2"]
        }
        
        # Forward pass
        with torch.no_grad():
            output = model.forward(batch)
            
        # Check output shape: [B, num_classes, D, H, W]
        expected_shape = (batch_size, 2, 32, 32, 32)
        if isinstance(output, list):  # Handle deep supervision
            output = output[0]
        self.assertEqual(output.shape, expected_shape)

    def test_segmentation_loss_computation(self):
        """Test segmentation loss computation"""
        config = self._build_segmentation_config()
        model = BaseSupervisedModel.create(task_type="segmentation", config=config)
        
        # Create dummy batch
        batch_size = 2
        image = torch.randn(batch_size, 3, 32, 32, 32)
        label = torch.randint(0, 2, (batch_size, 1, 32, 32, 32))  # Add channel dimension
        
        batch = {
            "image": image,
            "label": label,
            "file_path": ["test1", "test2"]
        }
        
        # Training step
        loss = model.training_step(batch, _batch_idx=0)
        
        # Check loss is computed
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() >= 0)  # Loss should be non-negative
        self.assertFalse(torch.isnan(loss))  # Loss should not be NaN

    def test_segmentation_metrics(self):
        """Test segmentation metrics computation"""
        config = self._build_segmentation_config()
        model = BaseSupervisedModel.create(task_type="segmentation", config=config)
        
        # Create dummy predictions and targets
        batch_size = 2
        num_classes = 2
        spatial_dims = (16, 16, 16)  # Smaller for faster testing
        
        # Create predictions (logits)
        predictions = torch.randn(batch_size, num_classes, *spatial_dims)
        
        # Create targets (ground truth segmentation)
        targets = torch.randint(0, num_classes, (batch_size, *spatial_dims))
        
        # Convert predictions to probabilities and class predictions
        probs = torch.softmax(predictions, dim=1)
        pred_classes = torch.argmax(probs, dim=1)
        
        # Test metrics computation
        val_metrics = model._configure_metrics("val")
        computed_metrics = model.compute_metrics(
            val_metrics, pred_classes, targets, ignore_index=0
        )
        
        # Check that metrics are computed
        self.assertIn("val/dice", computed_metrics)
        self.assertIn("val/f1", computed_metrics)
        
        # Check metric values are reasonable
        dice_value = computed_metrics["val/dice"]
        self.assertTrue(0 <= dice_value <= 1)

    def test_multi_encoder_segmentation(self):
        """Test multi-encoder setup for segmentation"""
        config = self._build_segmentation_config()
        config["use_multi_encoder"] = True
        
        model = BaseSupervisedModel.create(task_type="segmentation", config=config)
        
        # Check multi-encoder is properly configured
        self.assertTrue(hasattr(model.model.encoder, 'encoders'))
        
        # Test forward pass with modality mask
        batch_size = 1
        image = torch.randn(batch_size, 3, 32, 32, 32)
        label = torch.randint(0, 2, (batch_size, 32, 32, 32))
        modality_mask = torch.ones(batch_size, 3)  # All modalities present
        
        batch = {
            "image": image,
            "label": label,
            "modality_mask": modality_mask,
            "file_path": ["test"]
        }
        
        with torch.no_grad():
            output = model.forward(batch)
            
        # Check output shape
        if isinstance(output, list):
            output = output[0]
        self.assertEqual(output.shape, (batch_size, 2, 32, 32, 32))


if __name__ == "__main__":
    unittest.main()
