import torch
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError

from models.supervised_reg import SupervisedRegModel


def build_model(age_norm=True):
    cfg = {
        "task": "Task003_FOMO3",
        "task_id": 3,
        "task_type": "regression",
        "experiment": "unittest",
        "model_name": "unet_xl",
        "model_dimensions": "3D",
        "num_classes": 1,
        "num_modalities": 1,
        "patch_size": (8, 8, 8),
        "starting_filters": 8,
        "version_dir": ".",
        "cls_head_dropout_p": 0.0,
        "age_normalization": age_norm,
        "age_mean": 0.0,
        "age_std": 1.0,
        "compile": False,
    }
    return SupervisedRegModel(config=cfg, learning_rate=1e-3, do_compile=False)


def test_corr_invariance_to_scaling_without_trainer():
    torch.manual_seed(0)
    model = build_model(age_norm=False)

    # Fake batch outputs/targets (1D), two cases: y = x and y = 2x + 3
    x = torch.randn(64)
    y1 = x.clone()
    y2 = 2.0 * x + 3.0

    # Use the model's train metrics directly
    # First batch
    model.compute_metrics(model.train_metrics, output=y1, target=x)
    if hasattr(model, "pearson_train"):
        corr1 = model.pearson_train.compute().item()
        model.pearson_train.reset()
    else:
        # compute_metrics must create pearson_train lazily
        raise AssertionError("pearson_train not created by compute_metrics")

    # Second batch (scaled + biased), correlation should be identical
    model.compute_metrics(model.train_metrics, output=y2, target=x)
    corr2 = model.pearson_train.compute().item()

    # Correlation should be equal up to small tolerance
    assert abs(corr1 - corr2) < 1e-6


def test_metric_objects_created_and_resettable():
    torch.manual_seed(0)
    model = build_model(age_norm=True)

    # Create some normalized targets and arbitrary outputs
    t = torch.randn(32)
    o = t * 0.9 + 0.1
    metrics_train = MetricCollection({"train/mae": MeanAbsoluteError()})
    metrics_val = MetricCollection({"val/mae": MeanAbsoluteError()})

    _ = model.compute_metrics(metrics_train, output=o, target=t)
    _ = model.compute_metrics(metrics_val, output=o, target=t)

    assert hasattr(model, "pearson_train")
    assert hasattr(model, "pearson_val")

    # Ensure compute/reset works without exceptions
    _ = model.pearson_train.compute()
    model.pearson_train.reset()
    _ = model.pearson_val.compute()
    model.pearson_val.reset()
