#!/usr/bin/env bash
set -euo pipefail

BASE_SAVE_DIR=./runs/fomo1_k3_simple
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
    --epochs 500 --batch_size 2 --patch_size 128 --num_devices 1 --num_workers 2 --new_version \
    --k_folds ${K} --fold_index ${f} \
    --early_stop_patience 1000000 \
    --enhanced_early_stopping standard
done


