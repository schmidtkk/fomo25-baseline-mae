#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python src/finetune.py \
  --taskid 2 \
  --data_dir /mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune/fomo-task2 \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --dwi_ckpt   /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/dwi.ckpt \
  --flair_ckpt /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/flair.ckpt \
  --other_ckpt /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/other.ckpt \
  --precision 32-true \
  --epochs 150 --batch_size 1 --patch_size 24 --num_devices 1 --num_workers 8 --new_version \
  --split_method simple_train_val_split --split_param 0.6667


