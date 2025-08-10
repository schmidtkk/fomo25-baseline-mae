#!/bin/bash
# Loop through task IDs from 1 to 3
for taskid in $(seq 1 3); do
    python src/data/preprocess/run_preprocessing.py \
        --taskid="$taskid" \
        --source_path="/mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune/fomo-task$taskid"
    # Add any additional commands here if needed
done
