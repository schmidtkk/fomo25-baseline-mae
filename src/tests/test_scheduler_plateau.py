import unittest
import torch

from models.supervised_base import BaseSupervisedModel


class TestSchedulerPlateau(unittest.TestCase):
    def test_plateau_scheduler_config(self):
        cfg = {
            "task_type": "classification",
            "task": "Task001_FOMO1",
            "task_id": 1,
            "experiment": "unittest",
            "model_name": "unet_xl",
            "model_dimensions": "3D",
            "run_type": "finetune",
            "save_dir": ".",
            "train_data_dir": ".",
            "version_dir": ".",
            "version": 0,
            "ckpt_path": None,
            "pretrained_weights_path": None,
            "seed": 0,
            "num_classes": 2,
            "num_modalities": 4,
            "image_extension": ".npy",
            "labels": {0: "neg", 1: "pos"},
            "batch_size": 1,
            "learning_rate": 1e-4,
            "patch_size": (32, 32, 32),
            "precision": "32-true",
            "augmentation_preset": "none",
            "epochs": 2,
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
            "use_multi_encoder": True,
            "multi_encoder_modalities": ["DWI", "ADC", "T2FLAIR", "SWI_OR_T2STAR"],
            "lr_scheduler": "plateau",
            "plateau_factor": 0.5,
            "plateau_patience": 2,
            "plateau_threshold": 1e-3,
            "plateau_cooldown": 0,
            "plateau_min_lr": 1e-7,
        }
        m = BaseSupervisedModel.create(task_type="classification", config=cfg, learning_rate=1e-4)
        # Build dummy step to ensure components initialize
        x = torch.randn(1, 4, 32, 32, 32)
        y = torch.randint(0, 2, (1,))
        batch = {"image": x, "label": y, "file_path": "FOMO1_sub_0001"}
        m.on_validation_epoch_start()
        m.validation_step(batch, 0)
        opt_sched = m.configure_optimizers()
        # Ensure ReduceLROnPlateau is selected via dict API
        self.assertIsInstance(opt_sched, dict)
        self.assertIn("lr_scheduler", opt_sched)
        # Ensure correct monitor is used
        self.assertEqual(opt_sched["lr_scheduler"]["monitor"], "val/loss")


if __name__ == "__main__":
    unittest.main()


