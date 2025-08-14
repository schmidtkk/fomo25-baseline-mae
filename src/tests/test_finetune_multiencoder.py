import os
import sys
import tempfile
import unittest
import logging
import torch

from models.supervised_base import BaseSupervisedModel


class TestFinetuneMultiEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Route logging warnings to stdout so shell redirection captures them easily
        logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
    def _build_config(self):
        # Minimal config for a classification-like head
        return {
            "task": "Task001_FOMO1",
            "task_id": 1,
            "task_type": "classification",
            "experiment": "unittest",
            "model_name": "unet_xl",
            "model_dimensions": "3D",
            "run_type": "finetune",
            "save_dir": tempfile.gettempdir(),
            "train_data_dir": tempfile.gettempdir(),
            "version_dir": tempfile.gettempdir(),
            "version": 0,
            "ckpt_path": None,
            "pretrained_weights_path": None,
            "seed": 0,
            "num_classes": 2,
            # 4 canonical modalities for FOMO1: DWI, ADC, T2FLAIR, SWI/T2*
            "num_modalities": 4,
            "image_extension": ".npy",
            "allow_missing_modalities": True,
            "labels": {0: "neg", 1: "pos"},
            "batch_size": 1,
            "learning_rate": 1e-4,
            "patch_size": (32, 32, 32),
            "precision": "bf16-mixed",
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
            # Multi-encoder
            "use_multi_encoder": True,
            "multi_encoder_modalities": [
                "DWI",
                "ADC",
                "T2FLAIR",
                "SWI_OR_T2STAR",
            ],
            # Map finetune modalities to 5-group global vocab present in pretrain ckpts
            "modality_to_global_group": {
                "DWI": "dwi",
                "ADC": "dwi",
                "T2FLAIR": "flair",
                "SWI_OR_T2STAR": "other",
            },
            "global_vocab": ["t1", "t2", "flair", "dwi", "other"],
        }

    def _make_pseudo_pretrain_state(self, model: BaseSupervisedModel, mods_to_fill=None):
        """
        Create a pseudo state_dict that mimics per-modality encoder weights as if loaded
        from separate pretrains. Fill only the requested modalities (mods_to_fill) to simulate
        different availability scenarios.
        """
        if mods_to_fill is None:
            mods_to_fill = ["DWI", "T2FLAIR"]
        # Inspect model state dict keys to find encoder paths
        keys = list(model.state_dict().keys())
        # Example expected: 'model.encoder.encoders.DWI.in_conv.<...>.weight'
        pseudo = {}
        for k in keys:
            if any(k.startswith(f"model.encoder.encoders.{m}.") for m in mods_to_fill):
                shape = model.state_dict()[k].shape
                pseudo[k] = torch.randn(shape)
        return pseudo

    def test_forward_with_mask_and_pseudo_weights(self):
        config = self._build_config()
        model = BaseSupervisedModel.create(task_type="classification", config=config)
        model.eval()

        # Create pseudo weights and load (strict=False) to simulate partial per-modality loads
        pseudo_state = self._make_pseudo_pretrain_state(model)
        loaded = model.load_state_dict(pseudo_state, strict=False)
        self.assertGreaterEqual(loaded, 1)

        # Build dummy input [B,M,D,H,W] and mask [B,M]
        # Use 32 so that deepest encoder scale still has >1 spatial element (avoid InstanceNorm error)
        B, M, D, H, W = 2, config["num_modalities"], 32, 32, 32
        x = torch.randn(B, M, D, H, W)
        # Simulate missing ADC in sample 0 and missing SWI in sample 1
        mask = torch.ones(B, M)
        mask[0, 1] = 0  # ADC missing for sample 0
        mask[1, 3] = 0  # SWI/T2* missing for sample 1
        x[0, 1] = 0
        x[1, 3] = 0

        # Forward
        with torch.no_grad():
            logits = model.model(x, mask=mask)

        # Classification head outputs [B, num_classes]
        self.assertEqual(tuple(logits.shape), (B, config["num_classes"]))
        # Fusion gamma should align with 5 pretrain groups
        fusion0 = model.model.encoder.fusions[0]
        if getattr(fusion0, "use_gamma", False):
            self.assertEqual(fusion0.gamma.shape[0], 5)

    def test_multiple_modality_weight_combinations(self):
        combos = [
            [],
            ["DWI"],
            ["T2FLAIR"],
            ["ADC"],
            ["SWI_OR_T2STAR"],
            ["DWI", "T2FLAIR"],
            ["DWI", "ADC", "T2FLAIR", "SWI_OR_T2STAR"],
        ]
        for mods in combos:
            with self.subTest(mods=mods):
                config = self._build_config()
                model = BaseSupervisedModel.create(task_type="classification", config=config)
                model.eval()

                pseudo_state = self._make_pseudo_pretrain_state(model, mods_to_fill=mods)
                loaded = model.load_state_dict(pseudo_state, strict=False)
                if len(mods) == 0:
                    self.assertGreaterEqual(loaded, 0)
                else:
                    self.assertGreater(loaded, 0)

                B, M, D, H, W = 2, config["num_modalities"], 32, 32, 32
                x = torch.randn(B, M, D, H, W)
                mask = torch.ones(B, M)
                # simulate some missing
                mask[0, 1] = 0
                x[0, 1] = 0
                with torch.no_grad():
                    logits = model.model(x, mask=mask)
                self.assertEqual(tuple(logits.shape), (B, config["num_classes"]))
                # Verify modality group ids are in [0..4]
                gids = getattr(model.model.encoder, "modality_group_ids", None)
                self.assertIsNotNone(gids)
                self.assertTrue(all(0 <= g <= 4 for g in gids))


if __name__ == "__main__":
    unittest.main()


