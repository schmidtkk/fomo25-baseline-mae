import torch
from models.supervised_reg import SupervisedRegModel


def test_regression_metrics_include_corr_and_mae():
    config = {
        "num_classes": 1,
        "num_modalities": 2,
        "patch_size": (8, 8, 8),
        "model_name": "unet_b",
        "version_dir": ".",
        "task_type": "regression",
    }
    m = SupervisedRegModel(config=config)
    # Fake batch
    B = 4
    output = torch.randn(B, 1)
    target = torch.randn(B, 1)
    metrics = m._configure_metrics(prefix="val")
    res = m.compute_metrics(metrics, output, target)
    # Ensure keys exist
    assert "val/mae" in res and "val/corr" in res


