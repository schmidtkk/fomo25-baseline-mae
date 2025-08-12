#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python src/finetune.py \
  --taskid 3 \
  --data_dir /mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune/fomo-task3 \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --modality_mapping "T1=t1,T2=t2" \
  --t1_ckpt   /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/t1.ckpt \
  --t2_ckpt   /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/t2.ckpt \
  --precision 32-true \
  --epochs 200 --batch_size 2 --num_devices 1 --num_workers 8 --new_version \
  --split_method simple_train_val_split --split_param 0.6667


