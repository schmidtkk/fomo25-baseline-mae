import unittest
import torch

from models.supervised_cls import SupervisedClsModel


class TestValidationTTA(unittest.TestCase):
    def test_tta_flip_codes(self):
        cfg = {
            "num_classes": 2,
            "num_modalities": 4,
            "patch_size": (128, 128, 128),
            "model_name": "unet_xl",
            "version_dir": ".",
            "task_type": "classification",
            "val_tta_enable": True,
            "val_tta_views": 8,
        }
        m = SupervisedClsModel(config=cfg, learning_rate=1e-4)
        m.on_validation_epoch_start()
        self.assertTrue(m._val_tta_enable)
        self.assertEqual(len(m._tta_codes), 8)

    def test_tta_forward_average(self):
        cfg = {
            "num_classes": 2,
            "num_modalities": 4,
            "patch_size": (32, 32, 32),
            "model_name": "unet_xl",
            "version_dir": ".",
            "task_type": "classification",
            "val_tta_enable": True,
            "val_tta_views": 2,
        }
        m = SupervisedClsModel(config=cfg, learning_rate=1e-4)
        m.on_validation_epoch_start()
        x = torch.randn(1, 4, 32, 32, 32)
        y = torch.randint(0, 2, (1,))
        batch = {"image": x, "label": y, "file_path": "FOMO1_sub_0001"}
        # should run without error and accumulate
        m.validation_step(batch, 0)
        self.assertGreater(m._val_count, 0)


if __name__ == "__main__":
    unittest.main()


