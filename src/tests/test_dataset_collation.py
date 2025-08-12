import unittest
import torch


def collate_modalities(samples, finetune_modalities):
    """Simplified collation: from a list of dicts with per-modality tensors and mask to dense batch tensors."""
    B = len(samples)
    M = len(finetune_modalities)
    # assume all tensors same spatial size
    D, H, W = samples[0]["images"][finetune_modalities[0]].shape[-3:]
    x = torch.zeros(B, M, D, H, W, dtype=samples[0]["images"][finetune_modalities[0]].dtype)
    mask = torch.zeros(B, M, dtype=torch.float32)
    for b, s in enumerate(samples):
        for i, mod in enumerate(finetune_modalities):
            if mod in s["images"]:
                x[b, i] = s["images"][mod]
                mask[b, i] = 1.0
    return x, mask


class TestDatasetCollation(unittest.TestCase):
    def test_collation_with_missing_modalities(self):
        finetune_modalities = ["DWI", "ADC", "T2FLAIR", "SWI_OR_T2STAR"]
        D, H, W = 16, 16, 16
        # sample 0 missing ADC
        s0 = {
            "images": {
                "DWI": torch.randn(1, D, H, W),
                "T2FLAIR": torch.randn(1, D, H, W),
                "SWI_OR_T2STAR": torch.randn(1, D, H, W),
            }
        }
        # sample 1 missing SWI_OR_T2STAR
        s1 = {
            "images": {
                "DWI": torch.randn(1, D, H, W),
                "ADC": torch.randn(1, D, H, W),
                "T2FLAIR": torch.randn(1, D, H, W),
            }
        }
        x, mask = collate_modalities([s0, s1], finetune_modalities)
        self.assertEqual(tuple(x.shape), (2, 4, D, H, W))
        self.assertTrue(torch.allclose(mask, torch.tensor([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()


