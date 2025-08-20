import os
import json
import numpy as np
import nibabel as nib
from typing import Dict, List

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from data.task_configs import task1_config
from utils.utils import parallel_process


CANONICAL_MODALITIES = ["DWI", "ADC", "T2FLAIR", "SWI_OR_T2STAR"]


def _detect_canonical_files(session_path: str) -> Dict[int, str]:
    """
    Return mapping from canonical index to file path for present modalities.
    Canonical order: 0=DWI, 1=ADC, 2=T2FLAIR, 3=SWI_OR_T2STAR (prefer SWI over T2S).
    """
    modality_mapping: Dict[int, str] = {}
    for file in os.listdir(session_path):
        if not file.endswith(".nii.gz"):
            continue
        f = file.lower()
        if "dwi" in f:
            idx = 0
        elif "adc" in f:
            idx = 1
        elif "flair" in f:
            idx = 2
        elif ("swi" in f) or ("t2s" in f) or ("t2star" in f):
            idx = 3
        else:
            continue
        src = join(session_path, file)
        if idx == 3 and 3 in modality_mapping:
            # prefer SWI over T2S if both present
            prev = modality_mapping[3]
            if ("swi" in f) and ("swi" not in prev.lower()):
                modality_mapping[3] = src
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

        # label
        label_file = join(labels_dir, folder_name, "ses_1", "label.txt")
        if not os.path.exists(label_file):
            return f"Error: No label file found for {folder_name}"

        subject_id = folder_name.replace(".", "_")
        # Detect present modalities
        mapping = _detect_canonical_files(session_path)
        if len(mapping) == 0:
            return f"Warning: No canonical modalities found for {folder_name}"

        # Build ordered list of present nib images and names
        present_indices: List[int] = [i for i in range(len(CANONICAL_MODALITIES)) if i in mapping]
        present_names: List[str] = [CANONICAL_MODALITIES[i] for i in present_indices]
        images = [nib.load(mapping[i]) for i in present_indices]

        # Joint preprocessing for spatial alignment across present modalities
        preprocessed_images, _ = preprocess_case_for_training_without_label(
            images=images,
            normalization_operation=[pp_config["norm_op"] for _ in present_indices],
            allow_missing_modalities=False,
            crop_to_nonzero=pp_config["crop_to_nonzero"],
            target_spacing=pp_config.get("target_spacing", None),  # Use task-specific spacing
        )
        # Normalize return type across yucca versions:
        # Some versions return a list of arrays, others return a stacked ndarray.
        if isinstance(preprocessed_images, list):
            preprocessed_images = np.stack(preprocessed_images, axis=0)
        # In extreme cases with a single present modality, ensure channel dim exists
        if isinstance(preprocessed_images, np.ndarray) and preprocessed_images.ndim == 3:
            preprocessed_images = preprocessed_images[None, ...]
        # Expect shape: [N_present, D, H, W]
        assert preprocessed_images.shape[0] == len(present_indices), (
            f"Unexpected preprocessed shape {getattr(preprocessed_images, 'shape', None)} "
            f"for {folder_name}; expected first dim == {len(present_indices)}"
        )

        # Save per-modality arrays into subject folder
        subj_out_dir = join(target_preprocessed, f"{prefix}_{subject_id}")
        ensure_dir_exists(subj_out_dir)

        for ch, name in enumerate(present_names):
            out_path = join(subj_out_dir, f"{name}.npy")
            np.save(out_path, preprocessed_images[ch])

        # Save presence mask
        mask = {name: 1 if idx in mapping else 0 for idx, name in enumerate(CANONICAL_MODALITIES)}
        with open(join(subj_out_dir, "mask.json"), "w") as f:
            json.dump(mask, f)

        # Copy label
        np_label_dst = join(subj_out_dir, "label.txt")
        with open(label_file, "rb") as src, open(np_label_dst, "wb") as dst:
            dst.write(src.read())

        return f"Processed {folder_name}"

    except Exception as e:
        return f"Error processing {folder_name}: {str(e)}"


def convert_and_preprocess_task1_fusion(
    source_path: str,
    output_path: str,
    num_workers=None,
):
    """
    Preprocess Task 1 into per-modality files per subject with a presence mask.
    """
    pp_config = task1_config
    task_name = pp_config["task_name"]
    prefix = "FOMO1"

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
        process_subject, tasks, num_workers, desc="Processing subjects for Task 1 (fusion)"
    )

    print(f"Task 1 fusion preprocessing completed. Data saved to {target_preprocessed}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess FOMO1 into per-modality files for fusion finetune"
    )
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="data/preprocessed")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    convert_and_preprocess_task1_fusion(
        source_path=args.source_path, output_path=args.output_path, num_workers=args.num_workers
    )


