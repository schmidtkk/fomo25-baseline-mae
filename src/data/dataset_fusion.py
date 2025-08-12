import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List

from batchgenerators.utilities.file_and_folder_operations import join
from yucca.modules.data.augmentation.transforms.cropping_and_padding import CropPad
from yucca.modules.data.augmentation.transforms.formatting import NumpyToTorch


# Known canonical modality sets per task
FOMO1_MODALITIES = ["DWI", "ADC", "T2FLAIR", "SWI_OR_T2STAR"]
FOMO3_MODALITIES = ["T1", "T2"]


class FusionCLSDataset(Dataset):
    """
    Dataset for finetune fusion pipeline where each subject directory contains per-modality .npy files
    and an optional mask.json, plus label.txt.

    Expects samples as directory paths: <...>/Task001_FOMO1_fusion/FOMO1_<subject_id>
    Produces data_dict compatible with existing augmentation pipeline: 'image' is stacked [M,D,H,W].
    Missing modalities are zero-filled; model derives mask from zeros or reads 'mask' if needed.
    """

    def __init__(
        self,
        samples: List[str],
        patch_size: Tuple[int, int, int],
        composed_transforms=None,
        task_type: str = "classification",
        allow_missing_modalities: Optional[bool] = False,
        p_oversample_foreground: Optional[float] = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        # Supports both classification and regression finetune
        assert task_type in ("classification", "regression"), (
            f"Unsupported task_type '{task_type}' for FusionCLSDataset"
        )
        self.samples = samples
        self.patch_size = patch_size
        self.composed_transforms = composed_transforms
        self.allow_missing_modalities = allow_missing_modalities
        self.croppad = CropPad(patch_size=self.patch_size)
        self.to_torch = NumpyToTorch()
        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.samples)

    def _load_label(self, subject_dir: str) -> np.ndarray:
        label_path = join(subject_dir, "label.txt")
        # Classification labels are integers; regression labels are floats
        dtype = float if self.task_type == "regression" else int
        return np.loadtxt(label_path, dtype=dtype)

    def _load_mask(self, subject_dir: str) -> Optional[dict]:
        mask_path = join(subject_dir, "mask.json")
        if os.path.exists(mask_path):
            with open(mask_path, "r") as f:
                return json.load(f)
        return None

    def _resolve_modalities(self, subject_dir: str) -> List[str]:
        """
        Decide which canonical modality set to use for this subject.
        - If T1/T2 style files are present, use FOMO3 order [T1, T2].
        - Else, fall back to FOMO1 order [DWI, ADC, T2FLAIR, SWI_OR_T2STAR].
        """
        has_t1_t2 = any(
            os.path.exists(join(subject_dir, f"{m}.npy")) for m in FOMO3_MODALITIES
        )
        if has_t1_t2:
            return FOMO3_MODALITIES
        # Default to FOMO1 canonical if present
        has_fomo1 = any(
            os.path.exists(join(subject_dir, f"{m}.npy")) for m in FOMO1_MODALITIES
        )
        if has_fomo1:
            return FOMO1_MODALITIES
        raise AssertionError(
            f"No recognized modality files found in {subject_dir}. "
            f"Expected one or more of: {FOMO3_MODALITIES + FOMO1_MODALITIES}"
        )

    def _load_per_modality(self, subject_dir: str) -> Tuple[np.ndarray, List[str]]:
        modalities = self._resolve_modalities(subject_dir)
        # Determine reference shape from first present modality
        ref = None
        for mod in modalities:
            npy = join(subject_dir, f"{mod}.npy")
            if os.path.exists(npy):
                ref = np.load(npy, mmap_mode="r").shape
                break
        assert ref is not None, f"No modality files found in {subject_dir}"

        stacked = np.zeros((len(modalities),) + ref, dtype=np.float32)
        for i, mod in enumerate(modalities):
            npy = join(subject_dir, f"{mod}.npy")
            if os.path.exists(npy):
                arr = np.load(npy, mmap_mode="r")
                stacked[i] = arr.astype(np.float32, copy=False)
        return stacked, modalities

    def __getitem__(self, idx: int):
        subject_dir = self.samples[idx]
        assert os.path.isdir(subject_dir), f"Expected subject directory, got {subject_dir}"

        data, modalities = self._load_per_modality(subject_dir)
        label = self._load_label(subject_dir)

        data_dict = {
            "file_path": subject_dir,
            "image": data,
            "label": label,
        }

        metadata = {"foreground_locations": []}
        # Apply crop/pad and transforms
        data_dict["label"] = None
        data_dict = self.croppad(data_dict, metadata)
        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)
        data_dict["label"] = label
        return self.to_torch(data_dict)


