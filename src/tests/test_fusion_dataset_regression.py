import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from data.dataset_fusion import FusionCLSDataset


class TestFusionDatasetRegression(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a minimal subject folder with T1/T2 and a float label
        subj = os.path.join(self.tmpdir, "FOMO3_sub_0001")
        os.makedirs(subj, exist_ok=True)
        np.save(os.path.join(subj, "T1.npy"), np.zeros((8, 8, 8), dtype=np.float32))
        np.save(os.path.join(subj, "T2.npy"), np.ones((8, 8, 8), dtype=np.float32))
        with open(os.path.join(subj, "label.txt"), "w") as f:
            f.write("42\n")
        self.samples = [subj]

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_loads_t1_t2_and_float_label(self):
        ds = FusionCLSDataset(
            samples=self.samples,
            patch_size=(8, 8, 8),
            composed_transforms=None,
            task_type="regression",
        )
        item = ds[0]
        x, y = item["image"], item["label"]
        # Expect [M=2, D, H, W]
        self.assertEqual(tuple(x.shape), (2, 8, 8, 8))
        # Float label (accept numpy or torch)
        if isinstance(y, np.ndarray):
            self.assertTrue(np.issubdtype(y.dtype, np.floating))
        else:
            self.assertTrue(torch.is_floating_point(torch.as_tensor(y)))


if __name__ == "__main__":
    unittest.main()


