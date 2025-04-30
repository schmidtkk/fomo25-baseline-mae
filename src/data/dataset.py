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
import h5py
import logging


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
        import time
        # print(f"Loading {case} with shape {data.shape}")
        begin = time.time()
        data = self._load_volume(case)
        mid = time.time()
        data_dict = {
            "file_path": case
        }  # metadata that can be very useful for debugging.
        metadata = {"foreground_locations": []}

        data_dict["image"] = data
        res = self._transform(data_dict, metadata)
        end = time.time()
        print(f"Loaded {case} with shape {data.shape} in {end - begin:.2f} seconds (load: {mid - begin:.2f}, transform: {end - mid:.2f})")
        return res

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


# class PretrainDataset(Dataset):
#     def __init__(
#         self,
#         samples: list,
#         patch_size: Tuple[int, int, int],
#         h5_path: str,
#         pre_aug_patch_size: Optional[Tuple[int, int, int]] = None,
#         composed_transforms: Optional[torchvision.transforms.Compose] = None,
#     ):
#         """
#         Initialize the dataset with HDF5 file support.
        
#         Args:
#             samples: List of sample names (e.g., ['sample1', 'sample2'])
#             patch_size: Target patch size (D, H, W)
#             h5_path: Path to HDF5 file containing all samples
#             pre_aug_patch_size: Patch size before augmentation (optional)
#             composed_transforms: Optional torchvision transforms
#         """
#         self.all_files = samples
#         self.h5_path = h5_path
#         self.composed_transforms = composed_transforms
#         self.patch_size = patch_size
#         self.pre_aug_patch_size = pre_aug_patch_size

#         self.croppad = CropPad(patch_size=self.pre_aug_patch_size or self.patch_size)
#         self.to_torch = NumpyToTorch()

#         # Open HDF5 file in read mode
#         try:
#             print(f"Opening HDF5 file {h5_path}...")
#             self.h5_file = h5py.File(h5_path, 'r', libver='latest', swmr=True)
#             print(f"Successfully opened HDF5 file {h5_path}")
#         except Exception as e:
#             logging.error(f"Failed to open HDF5 file {h5_path}: {e}")
#             raise

#     def __len__(self):
#         return len(self.all_files)

#     def __getitem__(self, idx):
#         case = self.all_files[idx]
#         assert isinstance(case, str), f"Sample name must be string, got {type(case)}"

#         import time
#         begin = time.time()
#         # Load volume and header from HDF5
#         try:
#             data, header = self._load_volume_and_header(case)
#         except Exception as e:
#             logging.error(f"Error loading sample {case}: {e}")
#             raise
#         mid = time.time()

#         data_dict = {
#             "file_path": case
#         }
#         metadata = {"foreground_locations": header.get("foreground_locations", [])}

#         data_dict["image"] = data
#         res = self._transform(data_dict, metadata)
#         end = time.time()
#         logging.info(f"Loaded {case} in {end - begin:.2f} seconds (load: {mid - begin:.2f}, transform: {end - mid:.2f})")
#         return res

#     def _transform(self, data_dict, metadata=None):
#         data_dict = self.croppad(data_dict, metadata)
#         if self.composed_transforms is not None:
#             data_dict = self.composed_transforms(data_dict)
#         return self.to_torch(data_dict)

#     def _load_volume_and_header(self, case):
#         """Load volume (.npy) and header (.pkl) from HDF5 group"""
#         try:
#             group = self.h5_file[case]
#         except KeyError:
#             raise ValueError(f"Sample {case} not found in {self.h5_path}")

#         # Load volume (npy)
#         if "npy" in group:
#             vol = np.array(group["npy"])
#         else:
#             logging.warning(f"No npy data for {case}, using zero array")
#             vol = np.zeros(self.patch_size)

#         # Add channel dimension if needed
#         if len(vol.shape) == 3:
#             vol = vol[np.newaxis, ...]

#         # Load header (pkl)
#         header = {}
#         if "pkl" in group:
#             pkl_data = group["pkl"]
#             if pkl_data.attrs.get("type") == "pickle_dict":
#                 header = {k: np.array(v) if isinstance(v, np.ndarray) else v
#                           for k, v in pkl_data.items()}
#             elif pkl_data.attrs.get("type") == "pickle_numpy":
#                 header["data"] = np.array(pkl_data)
#             else:
#                 logging.warning(f"Unsupported pickle data type for {case}")
#         else:
#             logging.warning(f"No pkl data for {case}")

#         return vol, header

#     def __del__(self):
#         """Close HDF5 file when dataset is destroyed"""
#         if hasattr(self, 'h5_file'):
#             self.h5_file.close()