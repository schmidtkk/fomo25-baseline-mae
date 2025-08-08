#!/usr/bin/env python3
"""
Example usage of fomo4.py preprocessing for HBN dataset.

This script demonstrates how to preprocess the HBN dataset T1w images
for Task 4 (age regression).
"""

import os
import sys
sys.path.append('/home/weidongguo/workspace/fomo2025/baseline-codebase-main/src')

from src.data.preprocess.hbn import convert_and_preprocess_task4


def main():
    # Define paths
    source_path = "/data/weidong/hbn_dataset/raw"  # Directory containing T1w images
    tsv_file_path = "/data/weidong/hbn_dataset/vols/participants.tsv"  # TSV file with participant info
    output_path = "/home/weidongguo/workspace/fomo2025/baseline-codebase-main/data/preprocessed"  # Output directory for preprocessed data
    
    # Number of parallel workers (adjust based on your system)
    num_workers = 4
    
    print("Starting FOMO4 preprocessing for HBN dataset...")
    print(f"Source path: {source_path}")
    print(f"TSV file: {tsv_file_path}")
    print(f"Output path: {output_path}")
    print(f"Number of workers: {num_workers}")
    
    # Check if paths exist
    if not os.path.exists(source_path):
        print(f"Error: Source path does not exist: {source_path}")
        return
    
    if not os.path.exists(tsv_file_path):
        print(f"Error: TSV file does not exist: {tsv_file_path}")
        return
    
    # Run preprocessing
    try:
        convert_and_preprocess_task4(
            source_path=source_path,
            tsv_file_path=tsv_file_path,
            output_path=output_path,
            num_workers=num_workers
        )
        print("Preprocessing completed successfully!")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
