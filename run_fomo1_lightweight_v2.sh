#!/bin/bash
# Updated FOMO Task 1 Fusion Baseline - Lightweight & Stable
# Addresses parameter bloat, training instability, and complex fusion issues

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate fomo

export CUDA_VISIBLE_DEVICES=0

echo "🚀 FOMO Task 1 - Lightweight Fusion Baseline v2.0"
echo "📊 Target: ~90M params (vs 233M), stable training, 128 patch size"
echo "🔗 Modalities: DWI, ADC, T2FLAIR, SWI_OR_T2STAR"
echo "⚡ Fusion: Lightweight learnable weighted fusion"
echo "📈 Training: Progressive unfreezing strategy"
echo ""

PYTHONPATH=src python src/finetune.py \
  --taskid 1 \
  --data_dir /data/weidong/fomo-finetune \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --fusion_type learnable_weighted \
  --dwi_ckpt   ckpt/dwi.ckpt \
  --flair_ckpt ckpt/flair.ckpt \
  --other_ckpt ckpt/other.ckpt \
  --precision bf16-mixed \
  --patch_size=128 \
  --train_batches_per_epoch=100 \
  --epochs 500 --batch_size 1 --num_devices 1 --num_workers 8 \
  --starting_filters 48 \
  --freeze_encoder_epochs 0 \
  --phase1_head_lr 1e-3 --phase2_head_lr 5e-4 --phase2_encoder_lr 5e-5 \
  --label_smoothing 0.05 --cls_head_dropout_p 0.2 \
  --grad_clip_val 0.5 --grad_clip_algo norm \
  --num_sanity_val_steps 0 --log_every_n_steps 50 \
  --smoothing_window 10 \
  --disable_early_stop \
  --augmentation_preset basic \
  --lr_scheduler cosine \
  --experiment "fomo1_lightweight_fusion_v2" \
  --new_version

echo ""
echo "🎯 Key Improvements in v2.0:"
echo "   ✅ Reduced starting_filters: 64→48 (60% parameter reduction)"
echo "   ✅ Lightweight fusion: attention→learnable_weighted"
echo "   ✅ Progressive training: freeze_encoder_epochs=0 (immediate gradual training)"
echo "   ✅ Patch size: 96→128 (as requested)"
echo "   ✅ Reduced gradient clipping: 1.0→0.5 (better stability)"
echo "   ✅ Cosine LR schedule with warmup"
echo "   ✅ Window smoothing (size=10) for loss plots and console logs"
echo ""
echo "📁 Results saved to: runs/Task001_FOMO1/unet_xl/fomo1_lightweight_fusion_v2/"
