#!/bin/bash
# FOMO Task 1 Modality Ablation Study - Optimized for Infarct Detection
# Focuses on clinically relevant modality combinations based on stroke imaging principles
# Key insight: DWI+ADC is the gold standard for acute infarct detection

export CUDA_VISIBLE_DEVICES=0

echo "🔬 FOMO Task 1 Optimized Infarct Detection Ablation Study"
echo "=========================================================="
echo "🏥 Clinical Focus: Acute stroke/infarct detection"
echo "📊 Optimized based on stroke imaging best practices"
echo "🎯 Prioritizes DWI+ADC combinations (clinical gold standard)"
echo ""

# Common parameters
COMMON_ARGS="--taskid 1 \
  --data_dir /data/weidong/fomo-finetune \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --precision bf16-mixed \
  --patch_size=128 \
  --train_batches_per_epoch=100 \
  --epochs 500 --batch_size 1 --num_devices 1 --num_workers 8 \
  --starting_filters 64 \
  --freeze_encoder_epochs 3 --phase1_head_lr 1e-3 --phase2_head_lr 5e-4 --phase2_encoder_lr 5e-5 \
  --label_smoothing 0.1 --cls_head_dropout_p 0.3 \
  --grad_clip_val 1.0 --grad_clip_algo norm \
  --num_sanity_val_steps 0 --log_every_n_steps 25 \
  --disable_early_stop \
  --smoothing_window 10 \
  --augmentation_preset all"

echo "📊 Phase 1: Critical Single Modality Baselines (Infarct Detection)"
echo "-----------------------------------------------------------------"
echo "🏥 Note: DWI is the gold standard for acute infarct detection"
echo ""

# echo "🧪 Testing DWI only (primary infarct detector)..."
# PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
#   --enabled_modalities "DWI" \
#   --dwi_ckpt ckpt/dwi.ckpt \
#   --fusion_type masked_mean \
#   --experiment "ablation_dwi_only"

# echo "🧪 Testing ADC only (confirms restricted diffusion)..."
# PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
#   --enabled_modalities "ADC" \
#   --dwi_ckpt ckpt/dwi.ckpt \
#   --fusion_type masked_mean \
#   --experiment "ablation_adc_only"

# echo "🧪 Testing T2FLAIR only (chronic changes, context)..."
# PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
#   --enabled_modalities "T2FLAIR" \
#   --flair_ckpt ckpt/flair.ckpt \
#   --fusion_type masked_mean \
#   --experiment "ablation_flair_only"


echo ""
echo "📊 Phase 2: Critical Pairwise Combinations (Clinically Optimal)"
echo "---------------------------------------------------------------"
echo "🏥 Note: DWI+ADC is the gold standard combination for infarct detection"
echo ""

echo "🧪 Testing DWI + ADC (gold standard diffusion pair) - Channel Attention..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --enabled_modalities "DWI,ADC" \
  --dwi_ckpt ckpt/dwi.ckpt \
  --fusion_type channel_attention \
  --experiment "ablation_dwi_adc_chanatt"

echo "🧪 Testing DWI + ADC (gold standard) - Learnable Weighted..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --enabled_modalities "DWI,ADC" \
  --dwi_ckpt ckpt/dwi.ckpt \
  --fusion_type learnable_weighted \
  --experiment "ablation_dwi_adc_learnable"

echo "🧪 Testing DWI + T2FLAIR (acute + chronic context)..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --enabled_modalities "DWI,T2FLAIR" \
  --dwi_ckpt ckpt/dwi.ckpt --flair_ckpt ckpt/flair.ckpt \
  --fusion_type channel_attention \
  --experiment "ablation_dwi_flair"

echo ""
echo "📊 Phase 3: Three-Modality Optimal Combination"
echo "----------------------------------------------"
echo "🏥 Note: Adding T2FLAIR to DWI+ADC for comprehensive infarct assessment"
echo ""

echo "🧪 Testing DWI + ADC + T2FLAIR (comprehensive infarct detection) - Attention..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --enabled_modalities "DWI,ADC,T2FLAIR" \
  --dwi_ckpt ckpt/dwi.ckpt --flair_ckpt ckpt/flair.ckpt \
  --fusion_type attention \
  --experiment "ablation_dwi_adc_flair_att"

echo "🧪 Testing DWI + ADC + T2FLAIR (comprehensive) - Learnable Weighted..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --enabled_modalities "DWI,ADC,T2FLAIR" \
  --dwi_ckpt ckpt/dwi.ckpt --flair_ckpt ckpt/flair.ckpt \
  --fusion_type learnable_weighted \
  --experiment "ablation_dwi_adc_flair_learnable"

echo ""
echo "📊 Phase 4: All Modalities - Fusion Method Comparison"
echo "-----------------------------------------------------"
echo "🏥 Note: Testing all modalities with different fusion approaches"
echo ""

echo "🧪 Testing all modalities with attention fusion..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --dwi_ckpt ckpt/dwi.ckpt --flair_ckpt ckpt/flair.ckpt --other_ckpt ckpt/other.ckpt \
  --fusion_type attention \
  --experiment "ablation_all_attention"

echo "🧪 Testing all modalities with learnable weighted fusion..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --dwi_ckpt ckpt/dwi.ckpt --flair_ckpt ckpt/flair.ckpt --other_ckpt ckpt/other.ckpt \
  --fusion_type learnable_weighted \
  --experiment "ablation_all_learnable"

echo "🧪 Testing all modalities with hybrid attention fusion..."
PYTHONPATH=src python src/finetune.py $COMMON_ARGS \
  --dwi_ckpt ckpt/dwi.ckpt --flair_ckpt ckpt/flair.ckpt --other_ckpt ckpt/other.ckpt \
  --fusion_type hybrid_attention \
  --experiment "ablation_all_hybrid"

echo ""
echo "✅ Optimized Infarct Detection Ablation Study Completed!"
echo "========================================================="
echo "📈 Results organized by clinical relevance:"
echo ""
echo "🏥 CLINICAL PRIORITY RANKING for Infarct Detection:"
echo "   1. DWI+ADC (gold standard diffusion pair)"
echo "   2. DWI alone (primary acute infarct detector)" 
echo "   3. DWI+ADC+T2FLAIR (comprehensive assessment)"
echo "   4. All modalities (maximum information)"
echo ""
echo "🔬 KEY FINDINGS TO ANALYZE:"
echo "   - Does ADC significantly improve DWI-only performance?"
echo "   - Which fusion method works best for the DWI+ADC pair?"
echo "   - Does T2FLAIR add value beyond DWI+ADC?"
echo "   - Is learnable_weighted fusion competitive with attention methods?"
echo ""
echo "🎯 Recommended next steps:"
echo "   - Use best DWI+ADC configuration for production"
echo "   - Consider computational cost vs. performance gains"
echo "   - Validate findings on held-out test set"
