import os
import json
import numpy as np
import nibabel as nib
from typing import Dict, List

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.functional.preprocessing import preprocess_case_for_training_with_label
from data.task_configs import task2_config
from utils.utils import parallel_process


CANONICAL_MODALITIES = ["DWI", "T2FLAIR", "SWI_OR_T2STAR"]


def _detect_canonical_files(session_path: str) -> Dict[int, str]:
    """
    Return mapping from canonical index to file path for present modalities.
    Canonical order: 0=DWI, 1=T2FLAIR, 2=SWI_OR_T2STAR (prefer SWI over T2S).
    """
    modality_mapping: Dict[int, str] = {}
    for file in os.listdir(session_path):
        if not file.endswith(".nii.gz"):
            continue
        f = file.lower()
        if "dwi" in f:
            idx = 0
        elif "flair" in f:
            idx = 1
        elif ("swi" in f) or ("t2s" in f) or ("t2star" in f):
            idx = 2
        else:
            continue
        src = join(session_path, file)
        if idx == 2 and 2 in modality_mapping:
            # prefer SWI over T2S if both present
            prev = modality_mapping[2]
            if ("swi" in f) and ("swi" not in prev.lower()):
                modality_mapping[2] = src
        else:
            modality_mapping[idx] = src
    return modality_mapping


def process_subject(task_info):
    folder_name, source_path, labels_dir, pp_config, target_preprocessed, prefix = (
        task_info
    )

    try:
        images_dir = join(source_path, "preprocessed")
        session_path = join(images_dir, folder_name, "ses_1")
        if not os.path.isdir(session_path):
            return f"Error: {folder_name} is not a valid directory"

        # segmentation mask
        seg_file = join(labels_dir, folder_name, "ses_1", "seg.nii.gz")
        if not os.path.exists(seg_file):
            return f"Error: No seg file found for {folder_name}"

        subject_id = folder_name.replace(".", "_")
        # Detect present modalities
        mapping = _detect_canonical_files(session_path)
        if len(mapping) == 0:
            return f"Warning: No canonical modalities found for {folder_name}"

        present_indices: List[int] = [i for i in range(len(CANONICAL_MODALITIES)) if i in mapping]
        present_names: List[str] = [CANONICAL_MODALITIES[i] for i in present_indices]
        images = [nib.load(mapping[i]) for i in present_indices]
        seg = nib.load(seg_file)

        # Joint preprocessing to align modalities and label
        pre_img, pre_lab, _ = preprocess_case_for_training_with_label(
            images=images,
            label=seg,
            normalization_operation=[pp_config["norm_op"] for _ in present_indices],
            allow_missing_modalities=False,
            crop_to_nonzero=pp_config["crop_to_nonzero"],
            keep_aspect_ratio_when_using_target_size=pp_config["keep_aspect_ratio"],
        )
        # Normalize type across yucca versions
        if isinstance(pre_img, list):
            pre_img = np.stack(pre_img, axis=0)
        if isinstance(pre_img, np.ndarray) and pre_img.ndim == 3:
            pre_img = pre_img[None, ...]
        pre_lab = (np.asarray(pre_lab) > 0).astype(np.uint8)

        # Save per-modality arrays into subject folder
        subj_out_dir = join(target_preprocessed, f"{prefix}_{subject_id}")
        ensure_dir_exists(subj_out_dir)

        for ch, name in enumerate(present_names):
            out_path = join(subj_out_dir, f"{name}.npy")
            np.save(out_path, pre_img[ch])

        # Save presence mask
        mask = {name: 1 if idx in mapping else 0 for idx, name in enumerate(CANONICAL_MODALITIES)}
        with open(join(subj_out_dir, "mask.json"), "w") as f:
            json.dump(mask, f)

        # Save preprocessed segmentation as NIfTI
        # Use identity affine; downstream pipeline treats as array
        nib.save(nib.Nifti1Image(pre_lab.astype(np.int16), np.eye(4)), join(subj_out_dir, "mask.nii.gz"))

        return f"Processed {folder_name}"

    except Exception as e:
        return f"Error processing {folder_name}: {str(e)}"


def convert_and_preprocess_task2_fusion(
    source_path: str,
    output_path: str,
    num_workers=None,
):
    """
    Preprocess Task 2 into per-modality files per subject with a presence mask and binary seg mask.
    """
    pp_config = task2_config
    task_name = pp_config["task_name"]
    prefix = "FOMO2"

    labels_dir = join(source_path, "labels")
    images_dir = join(source_path, "preprocessed")
    target_preprocessed = join(output_path, f"{task_name}_fusion")
    ensure_dir_exists(target_preprocessed)

    folder_names = [
        f for f in os.listdir(images_dir) if os.path.isdir(join(images_dir, f, "ses_1"))
    ]
    assert len(folder_names) > 0, "Did not collect any subjects to preprocess."

    tasks = [
        (folder_name, source_path, labels_dir, pp_config, target_preprocessed, prefix)
        for folder_name in folder_names
    ]

    parallel_process(
        process_subject, tasks, num_workers, desc="Processing subjects for Task 2 (fusion)"
    )

    print(f"Task 2 fusion preprocessing completed. Data saved to {target_preprocessed}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess FOMO2 into per-modality files and a seg mask for fusion finetune"
    )
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="data/preprocessed")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    convert_and_preprocess_task2_fusion(
        source_path=args.source_path, output_path=args.output_path, num_workers=args.num_workers
    )


