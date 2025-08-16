#!/bin/bash
# FOMO Task 3 Brain Age Regression Ablation Study
# Systematically test different loss functions, normalization, and fusion approaches

export CUDA_VISIBLE_DEVICES=1

echo "🧠 FOMO Task 3 - Brain Age Regression Ablation Study"
echo "====================================================="
echo "🎯 Task: Systematic evaluation of regression approaches"
echo "📊 Focus: Loss functions, normalization, and fusion strategies"
echo "📈 Primary metric: Pearson Correlation & MAE (years)"
echo ""

# Common parameters for all experiments
COMMON_ARGS="--taskid 3 \
  --data_dir /data/weidong/fomo-finetune \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --modality_mapping "T1=t1,T2=t2" \
  --t1_ckpt ckpt/t1.ckpt --t2_ckpt ckpt/t2.ckpt \
  --precision bf16-mixed \
  --patch_size=128 \
  --train_batches_per_epoch=100 \
  --epochs 500 --batch_size 2 --num_devices 1 --num_workers 0 \
  --starting_filters 64 \
  --freeze_encoder_epochs 4 --phase1_head_lr 1e-3 --phase2_head_lr 3e-4 --phase2_encoder_lr 3e-5 \
  --label_smoothing 0.0 --cls_head_dropout_p 0.2 \
  --grad_clip_val 1.0 --grad_clip_algo norm \
  --num_sanity_val_steps 0 --log_every_n_steps 10 \
  --disable_early_stop \
  --augmentation_preset basic \
  --val_tta_enable --val_tta_views 4 --val_tta_offsets 3 \
  --smoothing_window 10"

echo "📊 Phase 1: Loss Function Comparison"
echo "------------------------------------"
echo "🧠 Evaluating different loss functions for brain age regression"
echo ""

echo "🧪 Testing MAE Loss (robust to outliers)..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --loss_type mae \
  --age_normalization --age_mean 61.87 --age_std 15.09 \
  --fusion_type attention \
  --experiment "ablation_mae_loss_normalized"

echo "🧪 Testing MSE Loss (standard regression)..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --loss_type mse \
  --age_normalization --age_mean 61.87 --age_std 15.09 \
  --fusion_type attention \
  --experiment "ablation_mse_loss_normalized"

echo "🧪 Testing Huber Loss (combines MAE+MSE benefits)..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --loss_type huber \
  --age_normalization --age_mean 61.87 --age_std 15.09 \
  --fusion_type attention \
  --experiment "ablation_huber_loss_normalized"

echo ""
echo "📊 Phase 2: Age Normalization Impact"
echo "------------------------------------"
echo "🧠 Testing impact of age normalization on training stability"
echo ""

echo "🧪 Testing MAE Loss WITHOUT normalization..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --loss_type mae \
  --fusion_type attention \
  --experiment "ablation_mae_loss_raw"

echo "🧪 Testing MSE Loss WITHOUT normalization..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --loss_type mse \
  --fusion_type attention \
  --experiment "ablation_mse_loss_raw"

echo ""
echo "📊 Phase 3: Single Modality Baselines"
echo "-------------------------------------"
echo "🧠 Evaluating individual modality contributions"
echo ""

echo "🧪 Testing T1-only (structural anatomy)..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --enabled_modalities "T1" \
  --loss_type mae \
  --age_normalization --age_mean 61.87 --age_std 15.09 \
  --fusion_type masked_mean \
  --experiment "ablation_t1_only"

echo "🧪 Testing T2-only (tissue contrast)..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --enabled_modalities "T2" \
  --loss_type mae \
  --age_normalization --age_mean 61.87 --age_std 15.09 \
  --fusion_type masked_mean \
  --experiment "ablation_t2_only"

echo ""
echo "📊 Phase 4: Fusion Strategy Comparison"
echo "--------------------------------------"
echo "🧠 Testing different fusion approaches for T1+T2"
echo ""

echo "🧪 Testing Attention Fusion (adaptive weighting)..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --loss_type mae \
  --age_normalization --age_mean 61.87 --age_std 15.09 \
  --fusion_type attention \
  --experiment "ablation_fusion_attention"

echo "🧪 Testing Learnable Weighted Fusion..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --loss_type mae \
  --age_normalization --age_mean 61.87 --age_std 15.09 \
  --fusion_type learnable_weighted \
  --experiment "ablation_fusion_learnable"

echo "🧪 Testing Channel Attention Fusion..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --loss_type mae \
  --age_normalization --age_mean 61.87 --age_std 15.09 \
  --fusion_type channel_attention \
  --experiment "ablation_fusion_channel"

echo ""
echo "✅ Brain Age Regression Ablation Study Completed!"
echo "================================================="
echo "📈 Results to analyze in ./runs/Task003_FOMO3/unet_xl/"
echo ""
echo "🧠 KEY COMPARISONS TO MAKE:"
echo "   1. Loss Functions: MAE vs MSE vs Huber"
echo "   2. Normalization: With vs without age normalization"
echo "   3. Modality Contribution: T1-only vs T2-only vs T1+T2"
echo "   4. Fusion Strategy: Attention vs Learnable vs Channel"
echo ""
echo "📊 METRICS TO COMPARE:"
echo "   • Pearson Correlation (primary brain age metric)"
echo "   • MAE in years (interpretable error)"
echo "   • R² score (explained variance)"
echo "   • Training stability (loss curves)"
echo ""
echo "🎯 EXPECTED FINDINGS:"
echo "   • MAE loss should be more robust than MSE"
echo "   • Age normalization should improve training stability"
echo "   • T1+T2 fusion should outperform single modalities"
echo "   • Attention fusion should be competitive with other methods"
echo ""
echo "� NOTE: Using dataset-specific normalization (mean=61.87, std=15.09)"
echo "�🔬 Use best configuration for final production model!"
