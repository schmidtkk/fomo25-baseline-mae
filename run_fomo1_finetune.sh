#!/usr/bin/env bash
set -euo pipefail

BASE_SAVE_DIR=./runs/fomo1_k3
K=3

for f in $(seq 0 $((K-1))); do
  PYTHONPATH=src python src/finetune.py \
    --taskid 1 \
    --data_dir /mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune \
    --save_dir "${BASE_SAVE_DIR}/fold${f}" \
    --model_name unet_xl \
    --fusion_mode fusion \
    --dwi_ckpt   /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/dwi.ckpt \
    --flair_ckpt /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/flair.ckpt \
    --t1_ckpt    /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/t1.ckpt \
    --t2_ckpt    /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/t2.ckpt \
    --other_ckpt /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/other.ckpt \
    --precision 32-true \
    --epochs 500 --batch_size 1 --patch_size 128 --num_devices 1 --num_workers 1 --new_version \
    --use_ema \
    --export_subject_probs \
    --k_folds ${K} --fold_index ${f} \
    --enable_enhanced_aggregation \
    --enable_training_visualization \
    --enhanced_early_stopping robust \
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --visualization_update_freq 2
done