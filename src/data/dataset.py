from curses import meta
import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from yucca.modules.data.augmentation.transforms.cropping_and_padding import CropPad
from yucca.modules.data.augmentation.transforms.formatting import NumpyToTorch

from batchgenerators.utilities.file_and_folder_operations import join


class CLSDataset(Dataset):
    def __init__(
        self,
        samples: list,
        patch_size: Tuple[int, int, int],
        composed_transforms: Optional[torchvision.transforms.Compose] = None,
        task_type: str = "classification",
        allow_missing_modalities: Optional[bool] = False,
        p_oversample_foreground: Optional[float] = 0,
    ):
        super().__init__()
        # for compatibility with the datamodule
        assert task_type == "classification"

        self.all_files = samples
        self.composed_transforms = composed_transforms

        self.patch_size = patch_size

        self.croppad = CropPad(patch_size=self.patch_size)
        self.to_torch = NumpyToTorch()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        case = self.all_files[idx]

        # single modality
        assert isinstance(case, str)

        data = self._load_volume(case)
        label = self._load_label(case)
        data_dict = {
            "file_path": case,
            "image": data,
            "label": label,
        }

        metadata = {"foreground_locations": []}

        return self._transform(data_dict, metadata)

    def _transform(self, data_dict, metadata=None):
        label = data_dict["label"]
        data_dict["label"] = None
        data_dict = self.croppad(data_dict, metadata)
        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)

        data_dict["label"] = label
        return self.to_torch(data_dict)

    def _load_volume_and_header(self, file):
        vol = self._load_volume(file)
        header = load_pickle(file[: -len(".npy")] + ".pkl")
        return vol, header

    def _load_label(self, file):
        file = file + ".txt"
        label = np.loadtxt(file, dtype=int)

        return label

    def _load_volume(self, file):
        file = file + ".npy"

        try:
            return np.load(file, "r")
        except ValueError:
            return np.load(file, allow_pickle=True)


class DummyCLSDataset(Dataset):
    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        n_channels: int = 1,
        n_classes: Optional[int] = 6,
        composed_transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.composed_transforms = composed_transforms

    def __len__(self):
        return 42

    def __getitem__(self, idx):
        return {
            "image": torch.randn(1, 1, *self.patch_size),
            "label": torch.randint(0, self.n_classes, (1,)),
            "file_path": "null",
        }


class PretrainDataset(Dataset):
    def __init__(
        self,
        samples: list,
        patch_size: Tuple[int, int, int],
        data_dir: str,
        pre_aug_patch_size: Optional[Tuple[int, int, int]] = None,
        composed_transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        self.all_files = samples
        self.data_dir = data_dir
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size
        self.pre_aug_patch_size = pre_aug_patch_size

        self.croppad = CropPad(patch_size=self.pre_aug_patch_size or self.patch_size)
        self.to_torch = NumpyToTorch()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        case = self.all_files[idx]

        # single modality
        assert isinstance(case, str)
        data = self._load_volume(case)
        data_dict = {
            "file_path": case
        }  # metadata that can be very useful for debugging.
        metadata = {"foreground_locations": []}

        data_dict["image"] = data

        return self._transform(data_dict, metadata)

    def _transform(self, data_dict, metadata=None):
        data_dict = self.croppad(data_dict, metadata)
        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)
        return self.to_torch(data_dict)

    def _load_volume_and_header(self, file):
        vol = self._load_volume(file)
        header = load_pickle(file[: -len(".npy")] + ".pkl")
        return vol, header

    def _load_volume(self, file):
        file = file + ".npy"
        path = join(self.data_dir, file)

        try:
            vol = np.load(path, "r")
        except ValueError:
            vol = np.load(path, allow_pickle=True)

        # Add channel dimension if it doesn't exist
        if len(vol.shape) == 3:
            vol = vol[np.newaxis, ...]

        return vol
