import logging
import torch
import lightning as pl
from typing import Literal, Optional, Tuple, List, Dict
from torch.utils.data import DataLoader, Sampler
from torchvision.transforms import Compose
from yucca.pipeline.configuration.split_data import SplitConfig
from yucca.functional.array_operations.matrix_ops import get_max_rotated_size
from yucca.modules.data.augmentation.transforms.Spatial import Spatial
from data.dataset import PretrainDataset


class PretrainDataModule(pl.LightningDataModule):
    """Lightning DataModule handling pretraining dataset with optional modality filtering."""

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        batch_size: int,
        num_workers: int,
        splits_config: SplitConfig,
        split_idx: int,
        train_data_dir: str,
        modality_mode: str = "all",
        train_sampler: Optional[Sampler] = None,
        val_sampler: Optional[Sampler] = None,
        composed_train_transforms: Optional[Compose] = None,
        composed_val_transforms: Optional[Compose] = None,
        prefetch_factor: int = 2,
        disable_memmap: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.splits_config = splits_config
        self.split_idx = split_idx
        self.train_data_dir = train_data_dir
        self.modality_mode = modality_mode.lower()
        self.composed_train_transforms = composed_train_transforms
        self.composed_val_transforms = composed_val_transforms
        self.pre_aug_patch_size = (
            get_max_rotated_size(patch_size)
            if augmentations_include_spatial(composed_train_transforms)
            else None
        )
        if self.pre_aug_patch_size is not None and not isinstance(self.pre_aug_patch_size, tuple):
            raise TypeError("pre_aug_patch_size must be a tuple or None")

        self.num_workers = (
            max(0, int(torch.get_num_threads() - 1))
            if num_workers is None
            else num_workers
        )
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.prefetch_factor = prefetch_factor
        self.disable_memmap = disable_memmap

        self.modality_stats: Dict = {}

    # -------------------- Modality Filtering Utilities -------------------- #
    @staticmethod
    def _extract_modality(sample_name: str) -> Optional[str]:
        import re
        base = sample_name.split("/")[-1]
        # Strip common extensions if present (robust to mixed representations in splits/tests)
        if "." in base:
            # remove only the last extension (e.g. .npy / .pkl / .txt)
            base_no_ext = base.rsplit('.', 1)[0]
        else:
            base_no_ext = base
        m = re.match(r"^sub_[^_]+_ses_[^_]+_([a-zA-Z0-9]+)(?:_.+)?$", base_no_ext)
        return m.group(1).lower() if m else None

    @classmethod
    def filter_samples_by_modality(cls, samples: List[str], modality_mode: str):
        modality_mode = modality_mode.lower()
        other_group = {"scan", "pd", "swi", "t2s"}
        if modality_mode == "all":
            kept = {}
            for s in samples:
                mod = cls._extract_modality(s) or "unknown"
                kept[mod] = kept.get(mod, 0) + 1
            return samples, {
                "total_original": len(samples),
                "total_filtered": len(samples),
                "kept_modalities": kept,
                "dropped_modalities": {},
            }
        allowed = other_group if modality_mode == "other" else {modality_mode}
        kept_samples: List[str] = []
        kept_counts: Dict[str, int] = {}
        dropped_counts: Dict[str, int] = {}
        for s in samples:
            mod = cls._extract_modality(s)
            key = mod if mod is not None else "unknown"
            if mod in allowed:
                kept_samples.append(s)
                kept_counts[key] = kept_counts.get(key, 0) + 1
            else:
                dropped_counts[key] = dropped_counts.get(key, 0) + 1
        stats = {
            "total_original": len(samples),
            "total_filtered": len(kept_samples),
            "kept_modalities": kept_counts,
            "dropped_modalities": dropped_counts,
        }
        return kept_samples, stats

    def setup(self, stage: Literal["fit", "test", "predict"]):  # type: ignore[override]
        assert stage == "fit"
        self.train_samples = self.splits_config.train(self.split_idx)
        self.val_samples = self.splits_config.val(self.split_idx)

        if self.modality_mode != "all":
            t_filtered, t_stats = self.filter_samples_by_modality(self.train_samples, self.modality_mode)
            v_filtered, v_stats = self.filter_samples_by_modality(self.val_samples, self.modality_mode)
            self.train_samples = t_filtered
            self.val_samples = v_filtered
            self.modality_stats = {"mode": self.modality_mode, "train": t_stats, "val": v_stats}
        else:
            _, t_stats = self.filter_samples_by_modality(self.train_samples, self.modality_mode)
            _, v_stats = self.filter_samples_by_modality(self.val_samples, self.modality_mode)
            self.modality_stats = {"mode": self.modality_mode, "train": t_stats, "val": v_stats}

        if len(self.train_samples) == 0:
            raise RuntimeError(f"No training samples left after modality filtering mode='{self.modality_mode}'.")
        if len(self.val_samples) == 0:
            logging.warning("Validation set empty after modality filtering.")

        self.train_dataset = PretrainDataset(
            self.train_samples,
            data_dir=self.train_data_dir,
            composed_transforms=self.composed_train_transforms,
            pre_aug_patch_size=self.pre_aug_patch_size,  # type: ignore
            patch_size=self.patch_size,
            disable_memmap=self.disable_memmap,
        )
        self.val_dataset = PretrainDataset(
            self.val_samples,
            data_dir=self.train_data_dir,
            composed_transforms=self.composed_val_transforms,
            patch_size=self.patch_size,
            disable_memmap=self.disable_memmap,
        )

    def train_dataloader(self):
        sampler = self.train_sampler(self.train_dataset) if self.train_sampler is not None else None
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            sampler=sampler,
            shuffle=sampler is None,
        )

    def val_dataloader(self):
        sampler = self.val_sampler(self.val_dataset) if self.val_sampler is not None else None
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            sampler=sampler,
        )


def augmentations_include_spatial(augmentations):
    if augmentations is None:
        return False
    for augmentation in augmentations.transforms:
        if isinstance(augmentation, Spatial):  # pragma: no cover
            return True
    return False
            # still compute distribution for transparency
