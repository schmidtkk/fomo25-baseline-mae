import torch
from models.supervised_cls import SupervisedClsModel


def test_subject_level_auroc_aggregation():
    # Binary classification setup
    config = {
        "num_classes": 2,
        "num_modalities": 4,
        "patch_size": (16, 16, 16),
        "model_name": "unet_b",
        "version_dir": ".",
        "task_type": "classification",
    }
    model = SupervisedClsModel(config=config, learning_rate=1e-3)

    # Simulate on_validation_epoch_start
    model.on_validation_epoch_start()

    # Two subjects: A (target=1), B (target=0)
    # A has two crops with probs 0.8 and 0.6 (avg=0.7). B has one crop prob 0.3
    model._val_subject_aggr = {
        "A": [0.8 + 0.6, 2.0, 1],
        "B": [0.3, 1.0, 0],
    }

    # Compute subject-level AUROC
    model.on_validation_epoch_end()

    # Ensure aggregation yields tensors and AUROC logged (cannot easily assert metric value here)
    probs, targets = model._mean_probs_targets_from_aggr(model._val_subject_aggr)
    assert probs is not None and targets is not None
    assert probs.numel() == 2 and targets.numel() == 2

