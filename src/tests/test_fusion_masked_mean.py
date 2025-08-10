import unittest
import torch

from models.fusion.masked_mean import MaskedMeanFusion3D


class TestMaskedMeanFusion3D(unittest.TestCase):
    def test_all_present_equals_mean_of_pre_aligned(self):
        B, M, C, D, H, W = 2, 3, 8, 8, 8, 8
        feats = [torch.randn(B, C, D, H, W) for _ in range(M)]
        mask = torch.ones(B, M)

        fusion = MaskedMeanFusion3D(C, num_modalities=M, use_gamma=False, use_null_token=False)
        # Compute expected: mean of pre-aligned features
        aligned = [fusion._pre_align(f) for f in feats]
        expected = torch.stack(aligned, dim=1).mean(dim=1)

        out = fusion(feats, mask)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5, rtol=1e-5))

    def test_single_present_equals_that_feature(self):
        B, M, C, D, H, W = 2, 3, 8, 8, 8, 8
        feats = [torch.randn(B, C, D, H, W) for _ in range(M)]
        mask = torch.zeros(B, M)
        mask[:, 1] = 1  # only modality 1 present

        fusion = MaskedMeanFusion3D(C, num_modalities=M, use_gamma=False, use_null_token=False)
        aligned_1 = fusion._pre_align(feats[1])

        out = fusion(feats, mask)
        self.assertTrue(torch.allclose(out, aligned_1, atol=1e-5, rtol=1e-5))

    def test_all_missing_is_stable(self):
        B, M, C, D, H, W = 2, 3, 8, 8, 8, 8
        feats = [torch.randn(B, C, D, H, W) for _ in range(M)]
        mask = torch.zeros(B, M)  # all missing

        fusion = MaskedMeanFusion3D(C, num_modalities=M, use_gamma=False, use_null_token=False)
        out = fusion(feats, mask)
        # With zeros mask and clamp_min(1), output should be near 0 after masking
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(out.abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()


