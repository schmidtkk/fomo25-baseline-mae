#!/usr/bin/env python
"""
Utilities for K-fold ensemble inference in FOMO Task 1.

This module provides helper functions to discover and load K-fold model checkpoints
for ensemble predictions.
"""

import os
import glob
from typing import List, Optional
import logging


def find_kfold_checkpoints(base_dir: str, pattern: str = "fomo1_kfold_fold*", 
                          checkpoint_name: str = "best.ckpt") -> List[str]:
    """
    Find K-fold checkpoint files in a directory structure.
    
    Args:
        base_dir: Base directory to search for fold experiments
        pattern: Pattern to match fold directories (default: "fomo1_kfold_fold*")  
        checkpoint_name: Name of checkpoint file to find (default: "best.ckpt")
        
    Returns:
        List of checkpoint paths sorted by fold number
        
    Example:
        >>> checkpoints = find_kfold_checkpoints("./runs/Task001_FOMO1/unet_xl/")
        >>> print(checkpoints)
        ['./runs/Task001_FOMO1/unet_xl/fomo1_kfold_fold0/version_0/checkpoints/best.ckpt',
         './runs/Task001_FOMO1/unet_xl/fomo1_kfold_fold1/version_0/checkpoints/best.ckpt',
         ...]
    """
    checkpoint_paths = []
    
    # Search for fold directories
    fold_pattern = os.path.join(base_dir, pattern)
    fold_dirs = glob.glob(fold_pattern)
    
    for fold_dir in sorted(fold_dirs):
        # Look for checkpoint in standard PyTorch Lightning structure
        checkpoint_candidates = [
            os.path.join(fold_dir, "version_0", "checkpoints", checkpoint_name),
            os.path.join(fold_dir, "checkpoints", checkpoint_name),
            os.path.join(fold_dir, checkpoint_name),
        ]
        
        for candidate in checkpoint_candidates:
            if os.path.exists(candidate):
                checkpoint_paths.append(candidate)
                break
        else:
            logging.warning(f"No checkpoint found in {fold_dir}")
    
    return sorted(checkpoint_paths)


def validate_kfold_checkpoints(checkpoint_paths: List[str]) -> bool:
    """
    Validate that all K-fold checkpoint files exist.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        
    Returns:
        True if all checkpoints exist, False otherwise
    """
    missing_checkpoints = []
    
    for path in checkpoint_paths:
        if not os.path.exists(path):
            missing_checkpoints.append(path)
    
    if missing_checkpoints:
        logging.error(f"Missing checkpoint files: {missing_checkpoints}")
        return False
    
    return True


def create_ensemble_checkpoint_string(checkpoint_paths: List[str]) -> str:
    """
    Create comma-separated checkpoint string for CLI usage.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        
    Returns:
        Comma-separated string of checkpoint paths
        
    Example:
        >>> paths = ["fold0.ckpt", "fold1.ckpt", "fold2.ckpt"]
        >>> ensemble_string = create_ensemble_checkpoint_string(paths)
        >>> print(ensemble_string)  
        "fold0.ckpt,fold1.ckpt,fold2.ckpt"
    """
    return ",".join(checkpoint_paths)


def discover_best_kfold_experiment(runs_dir: str = "./runs", 
                                   task: str = "Task001_FOMO1",
                                   model: str = "unet_xl") -> Optional[List[str]]:
    """
    Automatically discover the best K-fold experiment checkpoints.
    
    Args:
        runs_dir: Base runs directory
        task: Task name (default: "Task001_FOMO1")
        model: Model name (default: "unet_xl")
        
    Returns:
        List of checkpoint paths for the most recent K-fold experiment,
        or None if no K-fold experiments found
    """
    base_search_dir = os.path.join(runs_dir, task, model)
    
    if not os.path.exists(base_search_dir):
        logging.warning(f"Directory not found: {base_search_dir}")
        return None
    
    # Look for K-fold experiments
    kfold_experiments = []
    
    # Search for different K-fold patterns
    patterns = [
        "fomo1_kfold_fold*",
        "*kfold*fold*", 
        "fold*",
        "*_fold_*"
    ]
    
    for pattern in patterns:
        fold_dirs = glob.glob(os.path.join(base_search_dir, pattern))
        if fold_dirs:
            # Found fold directories with this pattern
            checkpoints = find_kfold_checkpoints(base_search_dir, 
                                                 os.path.basename(pattern))
            if len(checkpoints) >= 3:  # Require at least 3 folds
                kfold_experiments.append({
                    'pattern': pattern,
                    'checkpoints': checkpoints,
                    'count': len(checkpoints)
                })
    
    if not kfold_experiments:
        logging.warning(f"No K-fold experiments found in {base_search_dir}")
        return None
    
    # Return the experiment with the most folds
    best_experiment = max(kfold_experiments, key=lambda x: x['count'])
    logging.info(f"Found K-fold experiment with {best_experiment['count']} folds")
    
    return best_experiment['checkpoints']


def print_ensemble_usage_example():
    """Print usage examples for K-fold ensemble inference."""
    print("\n" + "="*60)
    print("FOMO Task 1 - K-Fold Ensemble Inference Examples")
    print("="*60)
    
    print("\n1. Manual checkpoint specification:")
    print("python3 -m inference.predict_task1 \\")
    print("  --dwi_b1000 /path/to/dwi.nii.gz \\")
    print("  --flair /path/to/flair.nii.gz \\")
    print("  --checkpoint /path/to/main/model.ckpt \\") 
    print("  --ensemble_checkpoints 'fold0.ckpt,fold1.ckpt,fold2.ckpt,fold3.ckpt,fold4.ckpt' \\")
    print("  --output /path/to/result.txt")
    
    print("\n2. Auto-discovery with TTA:")
    print("python3 -m inference.predict_task1 \\")
    print("  --dwi_b1000 /path/to/dwi.nii.gz \\")
    print("  --flair /path/to/flair.nii.gz \\")
    print("  --checkpoint /path/to/main/model.ckpt \\")
    print("  --tta_enable --tta_views 8 \\")
    print("  --output /path/to/result.txt")
    
    print("\n3. Full ensemble + TTA + uncertainty:")
    print("python3 -m inference.predict_task1 \\")
    print("  --dwi_b1000 /path/to/dwi.nii.gz \\")
    print("  --flair /path/to/flair.nii.gz \\")
    print("  --checkpoint /path/to/main/model.ckpt \\")
    print("  --ensemble_checkpoints 'fold0.ckpt,fold1.ckpt,fold2.ckpt,fold3.ckpt,fold4.ckpt' \\")
    print("  --tta_enable --tta_views 8 \\")
    print("  --confidence_threshold 0.8 \\")
    print("  --output /path/to/result.txt")
    
    print("\n4. Using discovery utility:")
    print("# In Python script:")
    print("from inference.kfold_ensemble_utils import discover_best_kfold_experiment")
    print("checkpoints = discover_best_kfold_experiment('./runs')")
    print("if checkpoints:")
    print("    ensemble_string = ','.join(checkpoints)")
    print("    print(f'Use --ensemble_checkpoints {ensemble_string}')")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Example usage when run as script
    print_ensemble_usage_example()
    
    # Try to discover K-fold experiments
    checkpoints = discover_best_kfold_experiment()
    if checkpoints:
        print(f"\nFound {len(checkpoints)} K-fold checkpoints:")
        for i, ckpt in enumerate(checkpoints):
            print(f"  Fold {i}: {ckpt}")
        
        ensemble_string = create_ensemble_checkpoint_string(checkpoints)
        print(f"\nCLI usage:")
        print(f"--ensemble_checkpoints '{ensemble_string}'")
    else:
        print("\nNo K-fold experiments found in ./runs/")
