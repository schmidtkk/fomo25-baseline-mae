import os
import pandas as pd
import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from data.task_configs import task4_config
from utils.utils import parallel_process


def process_subject(task_info):
    """
    Process a single subject for Task 4 (HBN dataset with T1w).

    Args:
        task_info: A tuple containing (participant_id, age, source_path, pp_config, target_preprocessed, prefix)

    Returns:
        Success message or error message
    """
    participant_id, age, source_path, pp_config, target_preprocessed, prefix = task_info

    try:
        # Extract subject identifier from participant_id (e.g., "sub-NDARAA075AMK" -> "NDARAA075AMK")
        subject_id = participant_id.replace("sub-", "")
        
        # Construct the T1w image file path
        image_file = f"{participant_id}_T1w.nii.gz"
        source_img = join(source_path, image_file)

        if not os.path.exists(source_img):
            return f"Error: Image file not found: {source_img}"

        # Load the T1w image
        img = nib.load(source_img)
        images = [img]  # Single modality

        # Apply preprocessing
        preprocessed_images, _ = preprocess_case_for_training_without_label(
            images=images,
            normalization_operation=[pp_config["norm_op"]],  # Single modality
            allow_missing_modalities=False,
            crop_to_nonzero=pp_config["crop_to_nonzero"],
        )

        # Save preprocessed data
        save_path = join(target_preprocessed, f"{prefix}_{subject_id}")
        np.save(save_path + ".npy", preprocessed_images)

        # Save age label for preprocessed data
        with open(join(target_preprocessed, f"{prefix}_{subject_id}.txt"), "w") as f:
            f.write(str(age))

        return f"Processed {participant_id}"

    except Exception as e:
        return f"Error processing {participant_id}: {str(e)}"


def convert_and_preprocess_task4(
    source_path: str,
    tsv_file_path: str,
    output_path: str,
    num_workers=None,
):
    """
    Preprocess all subjects for Task 4 (HBN dataset) in parallel.

    Args:
        source_path: Path to the raw data directory containing T1w images
        tsv_file_path: Path to the participants.tsv file
        output_path: Path where preprocessed data will be saved
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    # Get configuration from task4_config
    pp_config = task4_config
    task_name = pp_config["task_name"]
    prefix = "FOMO4"

    # Output path for preprocessed data
    target_preprocessed = join(output_path, task_name)

    # Create directory
    ensure_dir_exists(target_preprocessed)

    # Read the TSV file
    if not os.path.exists(tsv_file_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_file_path}")

    try:
        # Read CSV file (despite .tsv extension, it's comma-separated)
        df = pd.read_csv(tsv_file_path, sep=',')
        
        # Filter out rows with missing age data (NA values)
        df = df.dropna(subset=['Age'])
        
        print(f"Found {len(df)} subjects with valid age data")
        
    except Exception as e:
        raise ValueError(f"Error reading TSV file: {str(e)}")

    # Create task information for each subject
    tasks = []
    missing_files = []
    
    for _, row in df.iterrows():
        participant_id = row['participant_id']
        age = float(row['Age'])
        
        # Check if the T1w image file exists
        image_file = f"{participant_id}_T1w.nii.gz"
        source_img = join(source_path, image_file)
        
        if os.path.exists(source_img):
            tasks.append((participant_id, age, source_path, pp_config, target_preprocessed, prefix))
        else:
            missing_files.append(image_file)

    if missing_files:
        print(f"Warning: {len(missing_files)} image files not found:")
        for file in missing_files[:10]:  # Show first 10 missing files
            print(f"  - {file}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    print(f"Processing {len(tasks)} subjects for Task 4")

    # Process all subjects in parallel
    parallel_process(
        process_subject, tasks, num_workers, desc="Processing subjects for Task 4 (HBN)"
    )

    print(f"Task 4 preprocessing completed. Data saved to {target_preprocessed}")
    print(f"Successfully processed {len(tasks)} subjects")
