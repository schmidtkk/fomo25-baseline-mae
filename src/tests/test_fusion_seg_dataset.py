import os
import shutil
import tempfile
import unittest
import numpy as np
import nibabel as nib

from data.dataset_fusion import FusionDataset as FusionSegDataset


class TestFusionSegDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        subj = os.path.join(self.tmpdir, "FOMO2_sub_0001")
        os.makedirs(subj, exist_ok=True)
        # Create modalities and a simple spherical mask
        shape = (16, 16, 16)
        for name in ["DWI", "T2FLAIR", "SWI_OR_T2STAR"]:
            np.save(os.path.join(subj, f"{name}.npy"), np.random.randn(*shape).astype(np.float32))
        mask_np = np.zeros(shape, dtype=np.uint8)
        mask_np[4:12, 4:12, 4:12] = 1
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(mask_np.astype(np.int16), affine), os.path.join(subj, "mask.nii.gz"))
        self.samples = [subj]

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_loads_modalities_and_mask(self):
        ds = FusionSegDataset(
            samples=self.samples,
            patch_size=(16, 16, 16),
            composed_transforms=None,
            task_type="segmentation",
        )
        item = ds[0]
        x, y = item["image"], item["label"]
        self.assertEqual(tuple(x.shape), (3, 16, 16, 16))
        self.assertEqual(tuple(y.shape), (16, 16, 16))
        # Ensure y is binary
        self.assertTrue(set(np.unique(y.numpy()).tolist()) <= {0, 1})



