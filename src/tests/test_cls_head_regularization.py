import unittest
import torch

from models.supervised_cls import SupervisedClsModel


class TestClsHeadRegularization(unittest.TestCase):
    def test_label_smoothing_applied(self):
        cfg = {
            "num_classes": 2,
            "num_modalities": 4,
            "patch_size": (32, 32, 32),
            "model_name": "unet_xl",
            "version_dir": ".",
            "task_type": "classification",
            "label_smoothing": 0.1,
        }
        m = SupervisedClsModel(config=cfg, learning_rate=1e-4)
        self.assertAlmostEqual(m.loss_fn_train.label_smoothing, 0.1)

    def test_head_dropout_path(self):
        cfg = {
            "num_classes": 2,
            "num_modalities": 4,
            "patch_size": (32, 32, 32),
            "model_name": "unet_xl",
            "version_dir": ".",
            "task_type": "classification",
            "cls_head_dropout_p": 0.25,
        }
        m = SupervisedClsModel(config=cfg, learning_rate=1e-4)
        # Run a forward to initialize LazyLinear
        x = torch.randn(1, 4, 32, 32, 32)
        with torch.no_grad():
            _ = m(x)
        self.assertTrue(hasattr(m.model.decoder, "dropout"))
        self.assertAlmostEqual(getattr(m.model.decoder.dropout, "p", 0.0), 0.25)


if __name__ == "__main__":
    unittest.main()


