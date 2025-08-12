import os
import json
import tempfile

from models.supervised_cls import SupervisedClsModel


def test_writes_subject_probs_json_when_enabled():
    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "num_classes": 2,
            "num_modalities": 2,
            "patch_size": (8, 8, 8),
            "model_name": "unet_b",
            "version_dir": tmp,
            "task_type": "classification",
            "export_subject_probs": True,
        }
        model = SupervisedClsModel(config=config, learning_rate=1e-3)
        model.on_validation_epoch_start()
        model._val_subject_aggr = {
            "A": [0.8 + 0.6, 2.0, 1],
            "B": [0.3, 1.0, 0],
        }
        model.current_epoch = 0
        model.on_validation_epoch_end()

        out_dir = os.path.join(tmp, "subject_probs")
        assert os.path.isdir(out_dir)
        last_path = os.path.join(out_dir, "val_subject_probs_last.json")
        assert os.path.isfile(last_path)
        with open(last_path, "r") as f:
            data = json.load(f)
        assert set(data["subject_probs"].keys()) == {"A", "B"}
        assert data["epoch"] == 0


