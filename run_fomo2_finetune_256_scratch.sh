#!/bin/bash
# FOMO Task 2 Finetune Script - Meningioma Segmentation
# Task 2: Binary segmentation of brain meningiomas on MRI scans
# Modalities: DWI, T2FLAIR, SWI_OR_T2STAR
# Metrics: Dice Similarity Coefficient (DSC) & Normal Surface Distance (NSD)

export CUDA_VISIBLE_DEVICES=1

echo "🧠 FOMO Task 2 - Meningioma Segmentation"
echo "========================================"
echo "🎯 Task: Binary segmentation of brain meningiomas"
echo "👥 Cohort: 18+ years preoperative meningioma patients"
echo "📊 Modalities: DWI (b=1000), T2FLAIR, SWI/T2*"
echo "📈 Metrics: Dice Similarity Coefficient, Normal Surface Distance"
echo "🔗 Checkpoint mapping:"
echo "   DWI            → dwi.ckpt   (dwi group)"
echo "   T2FLAIR        → flair.ckpt (flair group)" 
echo "   SWI_OR_T2STAR  → other.ckpt (other group)"
echo ""

PYTHONPATH=src python3 src/finetune.py \
  --taskid 2 \
  --data_dir /data/weidong/fomo-finetune \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --fusion_type attention \
  --precision bf16-mixed \
  --patch_size 256,256,32 \
  --train_batches_per_epoch=100 \
  --epochs 300 --batch_size 1 --num_devices 1 --num_workers 8 \
  --starting_filters 64 \
  --freeze_encoder_epochs 3 --phase1_head_lr 1e-3 --phase2_head_lr 3e-4 --phase2_encoder_lr 3e-5 \
  --grad_clip_val 1.0 --grad_clip_algo norm \
  --num_sanity_val_steps 0 --log_every_n_steps 50 \
  --disable_early_stop \
  --augmentation_preset basic \
  --val_tta_enable --val_tta_views 4 --val_tta_offsets 3 \
  --smoothing_window 10 \
  --experiment "fomo2_meningioma_segmentation_256_256_32_scratch"

  # --dwi_ckpt   ckpt/dwi.ckpt \
  # --flair_ckpt ckpt/flair.ckpt \
  # --other_ckpt ckpt/other.ckpt \

# Segmentation-specific optimizations for Task 2:
# ===============================================
# 🎯 CRITICAL SETTINGS FOR MENINGIOMA SEGMENTATION:
#
# 1. **Task Type**: Automatically detected as segmentation from task2_config
#    - Uses SupervisedSegModel with Dice+CE loss
#    - Monitors val/dice for checkpoints 
#    - Computes segmentation-specific metrics (Dice, Jaccard, etc.)
#
# 2. **Model Architecture**:
#    - UNet-XL with segmentation head (2 classes: background + meningioma)
#    - Multi-encoder fusion for 3 modalities
#    - Attention-based fusion for better feature integration
#
# 3. **Loss Function**:
#    - DiceCE loss (Dice + Cross-entropy combination)
#    - Optimal for segmentation with class imbalance
#    - Handles small meningioma regions effectively
#
# 4. **Training Strategy**:
#    - 3 freeze epochs (shorter than classification tasks)
#    - Moderate learning rates for segmentation stability
#    - Basic augmentation to preserve spatial relationships
#    - TTA for robust validation estimates
#
# 5. **Evaluation Metrics**:
#    - Primary: Dice Similarity Coefficient (overlap-based)
#    - Secondary: Normal Surface Distance (boundary-based)
#    - Additional: Jaccard, Sensitivity, Precision
#
# 6. **Dataset Considerations**:
#    - Small dataset (23 finetune cases)
#    - Conservative training parameters to avoid overfitting
#    - Gradient clipping for training stability
#
# Expected output structure:
# runs/Task002_FOMO2/unet_xl/fomo2_meningioma_segmentation/version_0/
# ├── checkpoints/          # Model weights saved by Dice score
# ├── logs/                 # Training metrics and loss curves
# ├── best_metrics.txt      # Best validation metrics achieved
# └── predictions/          # Optional: validation set predictions

# Experiment naming examples:
# --experiment "fomo2_meningioma_segmentation"     (this run - baseline)
# --experiment "fomo2_dwi_only"                    (DWI-only ablation)
# --experiment "fomo2_no_dwi"                      (disable DWI)
# --experiment "fomo2_deeper_supervision"          (test deep supervision)
# --experiment "fomo2_augmentation_heavy"          (test stronger augmentations)
# --experiment "fomo2_patch_size_128"              (larger patches)
# --experiment "fomo2_freeze_epochs_5"             (longer freezing period)

# Key differences from classification tasks (Task 1, 3):
# - Segmentation loss (DiceCE) instead of classification/regression losses
# - Pixel-level predictions instead of subject-level predictions  
# - Dice coefficient monitoring instead of AUROC/correlation
# - Segmentation-specific augmentations and data loading
# - Surface distance metrics for boundary evaluation
