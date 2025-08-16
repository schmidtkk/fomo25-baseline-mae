#!/bin/bash
# FOMO Task 3 Finetune Script - Brain Age Regression
# Task 3: Predict patient age from T1w and T2w MRI scans
# Target: Healthy patients 18+ years old, metric: Absolute Error (AE) & Correlation

export CUDA_VISIBLE_DEVICES=0

echo "🧠 FOMO Task 3 - Brain Age Regression"
echo "====================================="
echo "🎯 Task: Predict patient age from MRI scans"
echo "📊 Modalities: T1w (structural) + T2w (anatomical detail)"
echo "👥 Cohort: Healthy patients 18+ years (no neurological conditions)"
echo "📈 Metrics: Absolute Error (AE), Correlation Coefficient"
echo "🔗 Checkpoint mapping:"
echo "   T1w → t1.ckpt   (structural brain anatomy)"
echo "   T2w → t2.ckpt   (tissue contrast & detail)"
echo ""

PYTHONPATH=src /home/weidongguo/miniconda3/envs/fomo/bin/python src/finetune.py \
  --taskid 3 \
  --data_dir /data/weidong/fomo-finetune \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --fusion_type attention \
  --modality_mapping "T1=t1,T2=t2" \
  --t1_ckpt   ckpt/t1.ckpt \
  --t2_ckpt   ckpt/t2.ckpt \
  --precision bf16-mixed \
  --patch_size=128 \
  --train_batches_per_epoch=100 \
  --epochs 500 --batch_size 2 --num_devices 1 --num_workers 0 \
  --starting_filters 64 \
  --freeze_encoder_epochs 4 --phase1_head_lr 1e-3 --phase2_head_lr 3e-4 --phase2_encoder_lr 3e-5 \
  --label_smoothing 0.0 \
  --cls_head_dropout_p 0.2 \
  --loss_type mae \
  --age_normalization \
  --age_mean 61.87 --age_std 15.09 \
  --grad_clip_val 1.0 --grad_clip_algo norm \
  --num_sanity_val_steps 0 \
  --disable_early_stop \
  --augmentation_preset basic \
  --val_tta_enable --val_tta_views 4 --val_tta_offsets 3 \
  --smoothing_window 10 \
  --experiment "fomo3_brain_age"

# Brain Age Regression Optimization Notes - ENHANCED:
# ====================================================
# 🧠 CRITICAL IMPROVEMENTS FOR BRAIN AGE REGRESSION:
# 
# 1. **Enhanced Loss Function**: 
#    - Using MAE loss (--loss_type mae) - more robust to age outliers than MSE
#    - MAE treats all age errors equally vs MSE which penalizes large errors disproportionately
#
# 2. **Age Normalization**: 
#    - Enabled age normalization (--age_normalization) for stable training
#    - Mean=50, Std=15 approximates typical brain age distribution
#    - Prevents gradient explosion from large raw age values (18-90+)
#
# 3. **Proper Metrics**: 
#    - Added Pearson correlation coefficient (primary brain age metric)
#    - MAE, MSE, R², and correlation all tracked
#    - Checkpoints saved based on val/pearson instead of val/loss
#
# 4. **Validation Strategy**:
#    - Primary metric changed from loss to correlation
#    - Better reflects actual model performance on brain age prediction
#    - More clinically relevant evaluation
#
# 5. **Training Parameters**:
#    - Conservative learning rates for stable age prediction
#    - Appropriate dropout (0.2) for regression head
#    - Basic augmentation to avoid overfitting small dataset
#
# 6. **Technical Optimizations**:
#    - bf16-mixed precision for efficiency
#    - Gradient clipping for training stability
#    - TTA for robust validation estimates

# Experiment naming examples:
# --experiment "fomo3_brain_age_enhanced"      (this run - enhanced regression)
# --experiment "fomo3_mse_loss"                (test MSE vs MAE loss)
# --experiment "fomo3_huber_loss"              (test Huber loss)
# --experiment "fomo3_no_normalization"        (disable age normalization)
# --experiment "fomo3_t1_only"                 (T1-only baseline)
# --experiment "fomo3_t2_only"                 (T2-only baseline) 
# --experiment "fomo3_learnable_weighted"      (test learnable weighted fusion)
# --experiment "fomo3_higher_lr"               (test learning rate sensitivity)

# Results organized by task in:
# runs/Task003_FOMO3/unet_xl/fomo3_brain_age_enhanced/version_0/
# Key metrics to track: MAE (years), Pearson Correlation, R²

# Enhanced regression improvements vs original classification approach:
# - MAE loss instead of MSE (robust to outliers)
# - Age normalization for training stability  
# - Pearson correlation as primary validation metric
# - Proper regression metrics (MAE in years, not loss units)
# - Age-aware validation strategy
# - Enhanced logging and monitoring for brain age tasks
# - Denormalized metrics for interpretable results (age in years)


