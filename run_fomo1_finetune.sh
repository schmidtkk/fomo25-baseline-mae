#!/bin/bash
# FOMO Task 1 Finetune Script - Optimized for actual modalities used
# Task 1 modalities: DWI, ADC, T2FLAIR, SWI_OR_T2STAR (NO T1/T2!)

export CUDA_VISIBLE_DEVICES=1

echo "🧬 FOMO Task 1 - Multi-encoder fusion training"
echo "📋 Using modalities: DWI, ADC, T2FLAIR, SWI_OR_T2STAR"
echo "🔗 Checkpoint mapping:"
echo "   DWI + ADC      → dwi.ckpt   (both use dwi group)"
echo "   T2FLAIR        → flair.ckpt (flair group)"
echo "   SWI_OR_T2STAR  → other.ckpt (other group)"
echo ""

PYTHONPATH=src python src/finetune.py \
  --taskid 1 \
  --data_dir /data/weidong/fomo-finetune \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --fusion_type attention \
  --dwi_ckpt   ckpt/dwi.ckpt \
  --flair_ckpt ckpt/flair.ckpt \
  --other_ckpt ckpt/other.ckpt \
  --precision bf16-mixed \
  --patch_size=96 \
  --train_batches_per_epoch=100 \
  --epochs 500 --batch_size 2 --num_devices 1 --num_workers 8 \
  --starting_filters 64 \
  --freeze_encoder_epochs 5 --phase1_head_lr 1e-3 --phase2_head_lr 5e-4 --phase2_encoder_lr 5e-5 \
  --label_smoothing 0.1 --cls_head_dropout_p 0.3 \
  --grad_clip_val 1.0 --grad_clip_algo norm \
  --num_sanity_val_steps 0 --log_every_n_steps 50 \
  --disable_early_stop \
  --augmentation_preset basic \
  --val_tta_enable --val_tta_views 4 --val_tta_offsets 3 \
  --smoothing_window 10 \
  --experiment "fomo1_optimized_baseline"
  
# Experiment naming examples:
# --experiment "fomo1_optimized_baseline"      (this run)
# --experiment "ablation_no_dwi"               (disable DWI with --disabled_modalities DWI)
# --experiment "attention_vs_mean_fusion"      (test different fusion types)
# --experiment "lr_schedule_experiment"        (test different learning rates)
# --experiment "freeze_epochs_sweep"           (test different freeze periods)

# Results will be saved in organized directories like:
# runs/Task001_FOMO1/unet_xl/fomo1_optimized_baseline/version_0/
# runs/Task001_FOMO1/unet_xl/ablation_no_dwi/version_0/
# etc.
  
# Removed unnecessary checkpoints: --t1_ckpt, --t2_ckpt (Task 1 doesn't use T1/T2!)
# Improved settings to address slow training loss:
# - fusion_type attention (better than simple averaging)
# - freeze_encoder_epochs 5 (was 1, too short)
# - higher phase1_head_lr 1e-3 (was 1e-4)
# - higher phase2_encoder_lr 5e-5 (was 2e-5)  
# - label_smoothing 0.1 (was 0.05, helps overfitting)
# - cls_head_dropout_p 0.3 (was 0.05, helps overfitting)
# - augmentation_preset basic (was all, less aggressive)
# - val_tta_enable for better validation estimates
  # ----new_version \
  # --val_batch_size 8 --accumulate_grad_batches 1 \
  # --lr_scheduler plateau --plateau_factor 0.5 --plateau_patience 6 --plateau_threshold 1e-3 --plateau_cooldown 0 --plateau_min_lr 1e-7 \
  # --val_batch_size 8 \
  # --val_tta_enable --val_tta_views 8 \
  # --val_tta_offsets 7 --val_tta_offset_frac 0.25 \
  # --val_tta_batch_size 8 \