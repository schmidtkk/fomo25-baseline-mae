import os
import json
import numpy as np
import nibabel as nib
from typing import Dict

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from data.task_configs import task3_config
from utils.utils import parallel_process


CANONICAL_MODALITIES = ["T1", "T2"]


def _detect_t1_t2_files(session_path: str) -> Dict[str, str]:
    """
    Return mapping from canonical modality name (T1/T2) to file path if present.
    Detection is based on case-insensitive filename contains 't1' or 't2'.
    """
    mapping: Dict[str, str] = {}
    for file in os.listdir(session_path):
        if not file.endswith(".nii.gz"):
            continue
        f = file.lower()
        if "t1" in f and "t2" not in f:
            mapping["T1"] = join(session_path, file)
        elif "t2" in f and "t1" not in f:
            mapping["T2"] = join(session_path, file)
    return mapping


def process_subject(task_info):
    folder_name, source_path, labels_dir, pp_config, target_preprocessed, prefix = (
        task_info
    )

    try:
        images_dir = join(source_path, "preprocessed")
        session_path = join(images_dir, folder_name, "ses_1")
        if not os.path.isdir(session_path):
            return f"Error: {folder_name} is not a valid directory"

        # label (age)
        label_file = join(labels_dir, folder_name, "ses_1", "label.txt")
        if not os.path.exists(label_file):
            return f"Error: No label file found for {folder_name}"

        # Detect present modalities
        mapping = _detect_t1_t2_files(session_path)
        present_names = [m for m in CANONICAL_MODALITIES if m in mapping]
        if len(present_names) == 0:
            return f"Warning: No T1/T2 modalities found for {folder_name}"

        # Load images
        images = [nib.load(mapping[name]) for name in present_names]

        # Joint preprocessing for spatial alignment
        preprocessed_images, _ = preprocess_case_for_training_without_label(
            images=images,
            normalization_operation=[pp_config["norm_op"] for _ in present_names],
            allow_missing_modalities=False,
            crop_to_nonzero=pp_config["crop_to_nonzero"],
            target_spacing=pp_config.get("target_spacing", None),  # Use task-specific spacing
        )

        # Normalize return type (yucca version differences)
        if isinstance(preprocessed_images, list):
            preprocessed_images = np.stack(preprocessed_images, axis=0)
        if isinstance(preprocessed_images, np.ndarray) and preprocessed_images.ndim == 3:
            preprocessed_images = preprocessed_images[None, ...]
        assert preprocessed_images.shape[0] == len(present_names), (
            f"Unexpected preprocessed shape {getattr(preprocessed_images, 'shape', None)} "
            f"for {folder_name}; expected first dim == {len(present_names)}"
        )

        # Save per-modality arrays into subject folder
        subj_out_dir = join(target_preprocessed, f"{prefix}_{folder_name.replace('.', '_')}")
        ensure_dir_exists(subj_out_dir)

        for ch, name in enumerate(present_names):
            out_path = join(subj_out_dir, f"{name}.npy")
            np.save(out_path, preprocessed_images[ch])

        # Save presence mask (optional)
        mask = {name: int(name in present_names) for name in CANONICAL_MODALITIES}
        with open(join(subj_out_dir, "mask.json"), "w") as f:
            json.dump(mask, f)

        # Copy label
        np_label_dst = join(subj_out_dir, "label.txt")
        with open(label_file, "rb") as src, open(np_label_dst, "wb") as dst:
            dst.write(src.read())

        return f"Processed {folder_name}"

    except Exception as e:
        return f"Error processing {folder_name}: {str(e)}"


def convert_and_preprocess_task3_fusion(
    source_path: str,
    output_path: str,
    num_workers=None,
):
    """
    Preprocess Task 3 (Brain Age) into per-modality files per subject with a presence mask.
    """
    pp_config = task3_config
    task_name = pp_config["task_name"]
    prefix = "FOMO3"

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
        process_subject, tasks, num_workers, desc="Processing subjects for Task 3 (fusion)"
    )

    print(f"Task 3 fusion preprocessing completed. Data saved to {target_preprocessed}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess FOMO3 into per-modality files for fusion finetune"
    )
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="data/preprocessed")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    convert_and_preprocess_task3_fusion(
        source_path=args.source_path, output_path=args.output_path, num_workers=args.num_workers
    )


