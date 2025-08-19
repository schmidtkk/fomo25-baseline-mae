#!/usr/bin/env python
"""
FOMO Task 2 Fusion Preprocessing

Task 2: Meningioma Segmentation
- Modalities: DWI, T2FLAIR, SWI_OR_T2STAR
- Labels: Binary segmentation masks (background=0, meningioma=1)
- Output: Per-modality .npy files + mask.json + seg.nii.gz

Based on fomo1_fusion.py and fomo3_fusion.py patterns
"""

import argparse
import logging
import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.functional.preprocessing import preprocess_case_for_training_with_label
from data.task_configs import task2_config


# Task 2 canonical modality order: DWI, T2FLAIR, SWI_OR_T2STAR
CANONICAL_MODALITIES = ("DWI", "T2FLAIR", "SWI_OR_T2STAR")


def process_subject(args):
    """
    Process a single subject for Task 2 fusion.
    
    Args:
        args: Tuple of (subject_id, source_path, output_path, task_config)
        
    Returns:
        str: Status message
    """
    subject_id, source_path, output_path, pp_config = args
    
    try:
        # Input paths
        images_dir = join(source_path, "preprocessed", subject_id, "ses_1")
        label_path = join(source_path, "labels", subject_id, "ses_1", "seg.nii.gz")
        
        if not os.path.exists(images_dir):
            return f"❌ Missing images directory: {images_dir}"
        
        if not os.path.exists(label_path):
            return f"❌ Missing segmentation label: {label_path}"
        
        # Output directory
        output_subject_dir = join(output_path, subject_id)
        ensure_dir_exists(output_subject_dir)
        
        # Find modality files in the images directory
        available_files = [f for f in os.listdir(images_dir) if f.endswith('.nii.gz')]
        
        # Map modalities to files
        modality_mapping: Dict[int, str] = {}
        
        for file in available_files:
            file_lower = file.lower()
            
            # Match DWI
            if any(pattern in file_lower for pattern in ['dwi', 'b1000']):
                modality_mapping[0] = join(images_dir, file)
            
            # Match T2FLAIR
            elif any(pattern in file_lower for pattern in ['flair', 't2flair']):
                modality_mapping[1] = join(images_dir, file)
            
            # Match SWI or T2STAR
            elif any(pattern in file_lower for pattern in ['swi', 't2s', 't2star', 't2*']):
                modality_mapping[2] = join(images_dir, file)
        
        # Check if we have all required modalities
        present_indices: List[int] = [i for i in range(len(CANONICAL_MODALITIES)) if i in modality_mapping]
        present_names: List[str] = [CANONICAL_MODALITIES[i] for i in present_indices]
        
        if len(present_indices) == 0:
            return f"❌ No recognized modalities found for {subject_id}"
        
        # Load images for preprocessing
        images = [nib.load(modality_mapping[i]) for i in present_indices]
        
        # Load segmentation label
        label = nib.load(label_path)
        
        # Apply preprocessing with segmentation label
        preprocessed_images, preprocessed_label, _ = preprocess_case_for_training_with_label(
            images=images,
            label=label,
            normalization_operation=[pp_config["norm_op"] for _ in present_indices],
            allow_missing_modalities=True,
            crop_to_nonzero=pp_config["crop_to_nonzero"],
            keep_aspect_ratio_when_using_target_size=pp_config["keep_aspect_ratio"],
        )
        
        # Validate preprocessing output
        if isinstance(preprocessed_images, list):
            preprocessed_images = np.array(preprocessed_images)
        
        assert preprocessed_images.shape[0] == len(present_indices), (
            f"Shape mismatch: preprocessed shape {preprocessed_images.shape} "
            f"for {subject_id}; expected first dim == {len(present_indices)}"
        )
        
        # Save per-modality .npy files
        full_modality_data = np.zeros((len(CANONICAL_MODALITIES), *preprocessed_images.shape[1:]))
        modality_mask = [0] * len(CANONICAL_MODALITIES)
        
        for i, present_idx in enumerate(present_indices):
            modality_name = CANONICAL_MODALITIES[present_idx]
            
            # Save individual modality file
            modality_path = join(output_subject_dir, f"{modality_name}.npy")
            np.save(modality_path, preprocessed_images[i])
            
            # Update full array and mask
            full_modality_data[present_idx] = preprocessed_images[i]
            modality_mask[present_idx] = 1
        
        # Save mask.json with presence flags
        mask_info = {
            "modalities": CANONICAL_MODALITIES,
            "mask": modality_mask,
            "present_modalities": present_names,
            "num_present": len(present_names)
        }
        with open(join(output_subject_dir, "mask.json"), "w") as f:
            json.dump(mask_info, f, indent=2)
        
        # Save segmentation label as .npy for training compatibility
        seg_path = join(output_subject_dir, "seg.npy")
        np.save(seg_path, preprocessed_label)
        
        # Also save original segmentation as .nii.gz for evaluation
        original_seg_path = join(output_subject_dir, "seg.nii.gz")
        seg_nii = nib.Nifti1Image(preprocessed_label.astype(np.uint8), affine=np.eye(4))
        nib.save(seg_nii, original_seg_path)
        
        return f"✅ Processed {subject_id} with modalities: {', '.join(present_names)}"
        
    except Exception as e:
        return f"❌ Error processing {subject_id}: {str(e)}"


def convert_and_preprocess_task2_fusion(
    source_path: str,
    output_path: str, 
    num_workers: Optional[int] = None,
):
    """
    Preprocess all subjects for Task 2 fusion in parallel.
    
    Args:
        source_path: Path to the source data directory
        output_path: Path where preprocessed data will be saved
        num_workers: Number of parallel workers (default: CPU count)
    """
    logging.info("FOMO Task 2 - Meningioma Segmentation Fusion Preprocessing")
    logging.debug("=" * 65)
    
    # Get task configuration
    pp_config = task2_config
    task_name = pp_config["task_name"]
    
    # Paths
    images_base_dir = join(source_path, "preprocessed")
    labels_base_dir = join(source_path, "labels")
    
    # Output directory with fusion suffix
    fusion_output_dir = join(output_path, f"{task_name}_fusion")
    ensure_dir_exists(fusion_output_dir)
    
    logging.info(f"Input images: {images_base_dir}")
    logging.info(f"Input labels: {labels_base_dir}")
    logging.info(f"Output directory: {fusion_output_dir}")
    logging.info(f"Target modalities: {', '.join(CANONICAL_MODALITIES)}")
    
    # Find all subjects
    if not os.path.exists(images_base_dir):
        raise ValueError(f"Images directory not found: {images_base_dir}")
    
    if not os.path.exists(labels_base_dir):
        raise ValueError(f"Labels directory not found: {labels_base_dir}")
    
    subjects = sorted([
        s for s in os.listdir(images_base_dir)
        if os.path.isdir(join(images_base_dir, s))
    ])
    
    logging.info(f"Found {len(subjects)} subjects to process")
    
    if len(subjects) == 0:
        raise ValueError("No subjects found for processing")
    
    # Prepare arguments for parallel processing
    process_args = [
        (subject, source_path, fusion_output_dir, pp_config)
        for subject in subjects
    ]
    
    # Process in parallel
    if num_workers is None:
        num_workers = min(len(subjects), os.cpu_count() or 1)
    
    logging.debug(f"Using {num_workers} workers for parallel processing")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_subject, args): args[0] 
            for args in process_args
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing subjects"):
            subject_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                # Keep per-subject progress minimal in console
                if not result.startswith("✅"):
                    tqdm.write(f"⚠️  {result}")
            except Exception as e:
                error_msg = f"❌ Failed to process {subject_id}: {str(e)}"
                results.append(error_msg)
                tqdm.write(error_msg)
    
    # Summary
    successful = len([r for r in results if r.startswith("✅")])
    failed = len(results) - successful
    
    logging.info("Processing Summary: success=%d, failed=%d, total=%d", successful, failed, len(subjects))
    logging.info("Task 2 fusion preprocessing completed. Output: %s", fusion_output_dir)
    logging.debug(f"Each subject contains: {', '.join(CANONICAL_MODALITIES)}.npy + mask.json + seg.npy")


def main():
    parser = argparse.ArgumentParser(
        description="FOMO Task 2 Meningioma Segmentation - Fusion Preprocessing"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to source fomo-task2 directory"
    )
    parser.add_argument(
        "--output_path", 
        type=str,
        required=True,
        help="Output parent directory (Task002_FOMO2_fusion will be created inside)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    
    args = parser.parse_args()
    
    convert_and_preprocess_task2_fusion(
        source_path=args.source_path,
        output_path=args.output_path,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
