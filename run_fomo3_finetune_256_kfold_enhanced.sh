#!/bin/bash
# FOMO Task 3 - 5-Fold Cross-Validation Brain Age Regression
# Enhanced script with age-stratified splits and comprehensive evaluation

set -e
export CUDA_VISIBLE_DEVICES=0,1

echo "🧠 FOMO Task 3 - 5-Fold Cross-Validation Brain Age Regression"
echo "=============================================================="
echo "🎯 Task: Systematic 5-fold evaluation of brain age prediction"
echo "📊 Modalities: T1w + T2w with attention fusion"
echo "📈 Metrics: Pearson Correlation, MAE (years), R²"
echo "🔬 Strategy: Age-stratified splits for balanced evaluation"
echo ""

# Configuration  
SCRIPT_DIR="$(cd "$(dirname "$0")" > /dev/null 2>&1 && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}"

# Experiment parameters
EXPERIMENT_BASE="fomo3_brain_age_256_kfold"
DATA_DIR="/data/weidong/fomo-finetune/Task003_FOMO3_fusion"
SAVE_DIR="./runs"
RESULTS_DIR="${SAVE_DIR}/Task003_FOMO3"
RESULTS_FILE="${RESULTS_DIR}/kfold_results_summary.txt"

# Training hyperparameters (optimized for brain age regression)
COMMON_ARGS="--taskid 3 \
  --data_dir ${DATA_DIR} \
  --save_dir ${SAVE_DIR} \
  --model_name unet_xl \
  --fusion_mode fusion \
  --fusion_type attention \
  --modality_mapping T1=t1,T2=t2 \
  --t1_ckpt ../ckpt/t1.ckpt \
  --t2_ckpt ../ckpt/t2.ckpt \
  --precision bf16-mixed \
  --patch_size=256,256,32 \
  --epochs 100 \
  --batch_size 2 \
  --num_devices 2 \
  --num_workers 4 \
  --starting_filters 64 \
  --freeze_encoder_epochs 4 \
  --phase1_head_lr 1e-3 \
  --phase2_head_lr 3e-4 \
  --phase2_encoder_lr 3e-5 \
  --label_smoothing 0.0 \
  --cls_head_dropout_p 0.2 \
  --loss_type mae \
  --age_normalization \
  --age_mean 61.87 \
  --age_std 15.09 \
  --grad_clip_val 1.0 \
  --grad_clip_algo norm \
  --num_sanity_val_steps 0 \
  --disable_early_stop \
  --augmentation_preset basic \
  --smoothing_window 10"

# Validation settings for robust evaluation
VALIDATION_ARGS="--val_tta_enable \
  --val_tta_views 4 \
  --val_tta_offsets 3 \
  --val_tta_offset_frac 0.25"

echo "📋 Training Configuration:"
echo "   Experiment: ${EXPERIMENT_BASE}"
echo "   Data Directory: ${DATA_DIR}"  
echo "   Results Directory: ${RESULTS_DIR}"
echo "   Patch Size: 256×256×32"
echo "   Loss Function: MAE (Mean Absolute Error)"
echo "   Age Normalization: μ=61.87, σ=15.09"
echo "   Test-Time Augmentation: Enabled"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Initialize results file
cat > "${RESULTS_FILE}" << EOF
FOMO Task 3 - 5-Fold Cross-Validation Results
Generated: $(date)
=============================================

Experimental Setup:
- Task: Brain Age Regression (Task003_FOMO3)
- Modalities: T1-weighted + T2-weighted MRI
- Architecture: UNet-XL with Multi-Modal Attention Fusion
- Cross-Validation: Age-Stratified 5-Fold
- Loss Function: Mean Absolute Error (MAE)
- Normalization: Z-score (age_mean=61.87, age_std=15.09)
- Test-Time Augmentation: 4 views, 3 offsets

Individual Fold Results:
========================

EOF

echo "🚀 Starting 5-fold cross-validation..."
echo ""

# Track timing and progress
START_TIME=$(date +%s)
TOTAL_FOLDS=5

# Run all 5 folds sequentially
for fold in 0 1 2 3 4; do
    FOLD_START_TIME=$(date +%s)
    fold_display=$((fold + 1))
    
    echo "🔄 Running Fold ${fold_display}/${TOTAL_FOLDS}..."
    echo "================================="
    echo "⏰ Started: $(date)"
    
    # Construct experiment name for this fold
    EXPERIMENT_NAME="${EXPERIMENT_BASE}_fold${fold}"
    
    # Run training for this fold
    echo "📚 Training with fold ${fold} as validation set..."
    
    # Execute training
    if cd task3 && python scripts/train_task3.py \
        $COMMON_ARGS \
        $VALIDATION_ARGS \
        --k_folds ${TOTAL_FOLDS} \
        --fold_index ${fold} \
        --experiment "${EXPERIMENT_NAME}"; then
        
        FOLD_END_TIME=$(date +%s)
        FOLD_DURATION=$((FOLD_END_TIME - FOLD_START_TIME))
        FOLD_HOURS=$((FOLD_DURATION / 3600))
        FOLD_MINUTES=$(((FOLD_DURATION % 3600) / 60))
        
        echo "✅ Completed Fold ${fold_display}/${TOTAL_FOLDS}"
        echo "⏱️  Duration: ${FOLD_HOURS}h ${FOLD_MINUTES}m"
        
        # Extract and log fold results immediately
        FOLD_RESULT_DIR="${SAVE_DIR}/Task003_FOMO3/unet_xl/${EXPERIMENT_NAME}/version_0"
        if [[ -d "${FOLD_RESULT_DIR}" ]]; then
            echo "📊 Extracting fold results..."
            
            # Try to extract key metrics from metrics.csv if available
            METRICS_FILE="${FOLD_RESULT_DIR}/metrics.csv"
            if [[ -f "${METRICS_FILE}" ]]; then
                # Extract best validation metrics using Python
                BEST_METRICS=$(python -c "
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('${METRICS_FILE}')
    
    results = {}
    # Correlation - higher is better
    if 'val/corr' in df.columns:
        val_corr = df['val/corr'].dropna()
        if len(val_corr) > 0:
            results['val_corr'] = val_corr.max()
    
    # MAE - lower is better  
    if 'val/mae' in df.columns:
        val_mae = df['val/mae'].dropna()
        if len(val_mae) > 0:
            results['val_mae'] = val_mae.min()
    
    # Print results
    for metric, value in results.items():
        print(f'{metric}: {value:.4f}')

except Exception as e:
    print(f'Error extracting metrics: {e}')
")
                
                if [[ -n "${BEST_METRICS}" ]]; then
                    echo "   ${BEST_METRICS}"
                    
                    # Append to results file
                    cat >> "${RESULTS_FILE}" << EOF
Fold ${fold} Results:
$(echo "${BEST_METRICS}" | sed 's/^/  /')
  Duration: ${FOLD_HOURS}h ${FOLD_MINUTES}m
  Completed: $(date)

EOF
                fi
            fi
        fi
        
    else
        echo "❌ Fold ${fold_display} failed!"
        echo "   Check logs for details"
        
        # Log failure
        cat >> "${RESULTS_FILE}" << EOF
Fold ${fold} Results:
  STATUS: FAILED
  Completed: $(date)

EOF
    fi
    
    echo ""
done

# Calculate total runtime
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo "📊 All folds completed! Computing aggregated metrics..."
echo "⏱️  Total Runtime: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"

# Aggregate results using our utility script
echo "🔍 Aggregating cross-validation results..."
if python src/utils/aggregate_kfold_results.py \
    --results_dir "${SAVE_DIR}/Task003_FOMO3/unet_xl" \
    --experiment_pattern "${EXPERIMENT_BASE}_fold*" \
    --output_file "${RESULTS_FILE}" \
    --metrics "val/corr,val/mae" \
    --verbose; then
    
    echo "✅ Results aggregation completed!"
else
    echo "⚠️  Results aggregation failed - manual review needed"
fi

# Final summary
cat >> "${RESULTS_FILE}" << EOF

Cross-Validation Summary:
========================
Total Runtime: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m
Completed: $(date)

Notes:
- Age-stratified splits ensure balanced age distribution across folds
- Test-time augmentation improves prediction robustness
- MAE loss optimized for brain age regression task
- Results suitable for publication-quality evaluation

EOF

echo ""
echo "🎉 5-Fold Cross-Validation Complete!"
echo "====================================="
echo "📈 Results summary: ${RESULTS_FILE}"
echo "📁 Individual fold checkpoints: ${SAVE_DIR}/Task003_FOMO3/unet_xl/"
echo "⏱️  Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo ""
echo "🧠 Key Metrics to Review:"
echo "   • Pearson Correlation (val/corr): Higher = Better brain age prediction"
echo "   • Mean Absolute Error (val/mae): Lower = More accurate age estimation"
echo "   • Cross-fold consistency: Lower std = More robust model"
echo ""
echo "📋 Next Steps:"
echo "   1. Review aggregated results in summary file"
echo "   2. Check individual fold performance for outliers"  
echo "   3. Consider ensemble prediction using all 5 folds"
echo "   4. Analyze age bias and demographic fairness"
