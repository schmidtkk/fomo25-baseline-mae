import os
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np


def _make_subject(dir_path: str, sid: str, label: int, shape=(16, 16, 16)):
    subj = os.path.join(dir_path, f"FOMO1_{sid}")
    os.makedirs(subj, exist_ok=True)
    for name in ["DWI", "ADC", "T2FLAIR", "SWI_OR_T2STAR"]:
        np.save(os.path.join(subj, f"{name}.npy"), np.random.randn(*shape).astype(np.float32))
    with open(os.path.join(subj, "mask.json"), "w") as f:
        json.dump({"DWI": 1, "ADC": 1, "T2FLAIR": 1, "SWI_OR_T2STAR": 1}, f)
    with open(os.path.join(subj, "label.txt"), "w") as f:
        f.write(str(int(label)))


def test_kfold_fusion_training_smoke():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    py = sys.executable

    with tempfile.TemporaryDirectory() as tmp:
        data_parent = tmp
        fusion_dir = os.path.join(data_parent, "Task001_FOMO1_fusion")
        os.makedirs(fusion_dir, exist_ok=True)

        # Create three small subjects to ensure non-empty train split for k=2
        # Use 32^3 so UNet downsampling keeps >1 spatial element
        _make_subject(fusion_dir, "A", label=1, shape=(32, 32, 32))
        _make_subject(fusion_dir, "B", label=0, shape=(32, 32, 32))
        _make_subject(fusion_dir, "C", label=1, shape=(32, 32, 32))

        save_dir = os.path.join(tmp, "runs")
        os.makedirs(save_dir, exist_ok=True)

        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env["PYTHONPATH"] = os.path.join(repo_root, "src")

        cmd = [
            py,
            os.path.join(repo_root, "src", "finetune.py"),
            "--taskid", "1",
            "--data_dir", data_parent,
            "--save_dir", save_dir,
            "--model_name", "unet_b",
            "--fusion_mode", "fusion",
            "--k_folds", "2",
            "--fold_index", "0",
            "--epochs", "1",
            "--train_batches_per_epoch", "1",
            "--batch_size", "1",
            "--num_devices", "1",
            "--num_workers", "0",
            "--precision", "32-true",
            "--augmentation_preset", "none",
            "--patch_size", "32",
            "--fast_dev_run",
            "--new_version",
        ]

        # Run training (short), expect success
        result = subprocess.run(cmd, env=env, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        assert result.returncode == 0, f"Training failed:\n{result.stdout}"

        # Check a version directory was created
        task_dir = os.path.join(save_dir, "Task001_FOMO1", "unet_b")
        assert os.path.isdir(task_dir), f"Missing task save dir: {task_dir}"

        # In fast_dev_run, Lightning suppresses checkpointing; we only smoke-test success



class TestKFoldFusionTraining(unittest.TestCase):
    def test_smoke(self):
        # Delegate to the standalone smoke test so it runs under unittest discovery
        test_kfold_fusion_training_smoke()
