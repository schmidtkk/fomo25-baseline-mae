#!/usr/bin/env python3
"""
Cross-Validation Utilities for FOMO Task3 Brain Age Regression

Provides age-stratified K-fold splitting for brain age regression tasks.
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import StratifiedKFold
import logging

logger = logging.getLogger(__name__)


def create_age_stratified_folds(subjects: List[str], ages: List[float], k: int = 5, 
                               random_state: int = 42) -> List[Tuple[List[str], List[str]]]:
    """
    Create K stratified folds based on age quartiles for brain age regression.
    
    For regression tasks, we use stratified binning to ensure balanced age 
    distribution across folds. This is crucial for brain age prediction to
    avoid age bias in train/val splits.
    
    Args:
        subjects: List of subject identifiers
        ages: List of corresponding ages (same length as subjects)
        k: Number of folds (default: 5)
        random_state: Random seed for reproducible splits
        
    Returns:
        List of (train_subjects, val_subjects) tuples, one per fold
        
    Example:
        >>> subjects = ["sub_001", "sub_002", "sub_003", "sub_004"]
        >>> ages = [25.0, 45.0, 65.0, 75.0]
        >>> folds = create_age_stratified_folds(subjects, ages, k=2)
        >>> train_subs, val_subs = folds[0]  # First fold
    """
    if len(subjects) != len(ages):
        raise ValueError(f"Subjects ({len(subjects)}) and ages ({len(ages)}) must have same length")
    
    if k < 2:
        raise ValueError(f"Number of folds must be >= 2, got {k}")
    
    if len(subjects) < k:
        raise ValueError(f"Need at least {k} subjects for {k}-fold CV, got {len(subjects)}")
    
    # Create age quartile bins for stratification
    ages_array = np.array(ages)
    age_quartiles = np.percentile(ages_array, [25, 50, 75])
    
    def get_age_bin(age: float) -> int:
        """Assign age to quartile bin (0=Q1, 1=Q2, 2=Q3, 3=Q4)."""
        if age <= age_quartiles[0]:
            return 0  # Q1: youngest quartile
        elif age <= age_quartiles[1]: 
            return 1  # Q2: second quartile
        elif age <= age_quartiles[2]:
            return 2  # Q3: third quartile
        else:
            return 3  # Q4: oldest quartile
    
    # Assign each subject to an age bin
    age_bins = [get_age_bin(age) for age in ages]
    
    # Log stratification distribution
    bin_counts = np.bincount(age_bins, minlength=4)
    logger.info(f"Age distribution across quartiles: Q1={bin_counts[0]}, Q2={bin_counts[1]}, "
               f"Q3={bin_counts[2]}, Q4={bin_counts[3]}")
    
    # Use StratifiedKFold with age bins
    stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(stratified_kfold.split(subjects, age_bins)):
        train_subjects = [subjects[i] for i in train_idx]
        val_subjects = [subjects[i] for i in val_idx]
        
        # Log fold statistics
        train_ages = [ages[i] for i in train_idx]
        val_ages = [ages[i] for i in val_idx]
        
        logger.info(f"Fold {fold_idx}: Train N={len(train_subjects)} (age: {np.mean(train_ages):.1f}±{np.std(train_ages):.1f}), "
                   f"Val N={len(val_subjects)} (age: {np.mean(val_ages):.1f}±{np.std(val_ages):.1f})")
        
        folds.append((train_subjects, val_subjects))
    
    return folds


def validate_kfold_splits(folds: List[Tuple[List[str], List[str]]]) -> Dict[str, float]:
    """
    Validate K-fold splits for quality and balance.
    
    Args:
        folds: List of (train_subjects, val_subjects) tuples
        
    Returns:
        Dictionary with validation statistics
    """
    k = len(folds)
    all_subjects = set()
    fold_sizes = []
    
    for fold_idx, (train_subjects, val_subjects) in enumerate(folds):
        # Check for overlap
        train_set = set(train_subjects)
        val_set = set(val_subjects)
        
        if train_set & val_set:
            overlap = train_set & val_set
            raise ValueError(f"Fold {fold_idx}: Train/val overlap detected: {overlap}")
        
        # Track subjects and sizes
        all_subjects.update(train_set)
        all_subjects.update(val_set)
        fold_sizes.append(len(val_subjects))
    
    # Compute statistics
    total_subjects = len(all_subjects)
    mean_val_size = np.mean(fold_sizes)
    std_val_size = np.std(fold_sizes)
    
    validation_stats = {
        'total_subjects': total_subjects,
        'num_folds': k,
        'mean_val_fold_size': mean_val_size,
        'std_val_fold_size': std_val_size,
        'val_fold_size_cv': std_val_size / mean_val_size if mean_val_size > 0 else 0,
    }
    
    logger.info(f"K-fold validation: {k} folds, {total_subjects} subjects, "
               f"val fold size: {mean_val_size:.1f}±{std_val_size:.1f}")
    
    return validation_stats


def load_subjects_and_ages_from_dataset(data_dir: str, task_id: int = 3) -> Tuple[List[str], List[float]]:
    """
    Load subject IDs and ages from FOMO dataset directory.
    
    Args:
        data_dir: Path to FOMO dataset directory (e.g., /data/weidong/fomo-finetune)
        task_id: Task ID (3 for Task003_FOMO3)
        
    Returns:
        Tuple of (subject_list, age_list)
    """
    task_dir = os.path.join(data_dir, f"Task{task_id:03d}_FOMO{task_id}")
    
    if not os.path.exists(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    
    # Look for labels file or dataset.json
    labels_file = os.path.join(task_dir, "labels.txt")
    dataset_json = os.path.join(task_dir, "dataset.json")
    
    subjects = []
    ages = []
    
    if os.path.exists(labels_file):
        # Load from labels.txt format: subject_id,age
        logger.info(f"Loading subjects and ages from: {labels_file}")
        with open(labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        subjects.append(parts[0].strip())
                        ages.append(float(parts[1].strip()))
    
    elif os.path.exists(dataset_json):
        # Load from nnUNet-style dataset.json
        import json
        logger.info(f"Loading subjects and ages from: {dataset_json}")
        
        with open(dataset_json, 'r') as f:
            dataset_info = json.load(f)
        
        # Extract from training data
        if 'training' in dataset_info:
            for item in dataset_info['training']:
                if 'image' in item and 'label' in item:
                    # Extract subject ID from image path
                    image_path = item['image']
                    subject_id = os.path.basename(image_path).split('_')[0]  # Extract subject prefix
                    subjects.append(subject_id)
                    ages.append(float(item['label']))
    
    else:
        raise FileNotFoundError(f"No labels file found in {task_dir}. "
                              f"Expected {labels_file} or {dataset_json}")
    
    if not subjects:
        raise ValueError(f"No subjects loaded from {task_dir}")
    
    logger.info(f"Loaded {len(subjects)} subjects with ages from {task_dir}")
    logger.info(f"Age range: {min(ages):.1f} - {max(ages):.1f} years (mean: {np.mean(ages):.1f}±{np.std(ages):.1f})")
    
    return subjects, ages


def save_fold_splits(folds: List[Tuple[List[str], List[str]]], output_dir: str, 
                     experiment_name: str = "kfold_splits") -> str:
    """
    Save K-fold splits to files for reproducibility.
    
    Args:
        folds: List of (train_subjects, val_subjects) tuples
        output_dir: Directory to save split files
        experiment_name: Base name for split files
        
    Returns:
        Path to the splits directory
    """
    splits_dir = os.path.join(output_dir, f"{experiment_name}_splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    # Save each fold split
    for fold_idx, (train_subjects, val_subjects) in enumerate(folds):
        fold_dir = os.path.join(splits_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save train subjects
        with open(os.path.join(fold_dir, "train_subjects.txt"), 'w') as f:
            for subject in train_subjects:
                f.write(f"{subject}\n")
        
        # Save validation subjects  
        with open(os.path.join(fold_dir, "val_subjects.txt"), 'w') as f:
            for subject in val_subjects:
                f.write(f"{subject}\n")
    
    # Save summary
    summary_file = os.path.join(splits_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"K-Fold Cross-Validation Splits: {experiment_name}\n")
        f.write(f"Generated: {np.datetime64('now')}\n")
        f.write(f"Number of folds: {len(folds)}\n")
        f.write("\n")
        
        for fold_idx, (train_subjects, val_subjects) in enumerate(folds):
            f.write(f"Fold {fold_idx}:\n")
            f.write(f"  Train: {len(train_subjects)} subjects\n")  
            f.write(f"  Val: {len(val_subjects)} subjects\n")
            f.write("\n")
    
    logger.info(f"Saved K-fold splits to: {splits_dir}")
    return splits_dir


if __name__ == "__main__":
    """Command-line interface for testing cross-validation utilities."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FOMO cross-validation utilities")
    parser.add_argument("--data_dir", type=str, 
                       default="/data/weidong/fomo-finetune",
                       help="Path to FOMO dataset directory")
    parser.add_argument("--task_id", type=int, default=3,
                       help="Task ID (3 for Task003_FOMO3)")
    parser.add_argument("--k_folds", type=int, default=5,
                       help="Number of folds")
    parser.add_argument("--output_dir", type=str, default="./test_kfold_splits",
                       help="Output directory for splits")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load subjects and ages
        print(f"🔍 Loading subjects and ages from Task{args.task_id:03d}...")
        subjects, ages = load_subjects_and_ages_from_dataset(args.data_dir, args.task_id)
        
        # Create stratified folds
        print(f"📊 Creating {args.k_folds}-fold stratified splits...")
        folds = create_age_stratified_folds(subjects, ages, k=args.k_folds, 
                                           random_state=args.random_state)
        
        # Validate splits
        print("✅ Validating K-fold splits...")
        validation_stats = validate_kfold_splits(folds)
        
        # Save splits
        print(f"💾 Saving splits to {args.output_dir}...")
        splits_dir = save_fold_splits(folds, args.output_dir, f"task{args.task_id}_kfold")
        
        print("🎉 Cross-validation splits generated successfully!")
        print(f"   Splits saved to: {splits_dir}")
        print(f"   Validation CV: {validation_stats['val_fold_size_cv']:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
