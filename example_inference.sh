#!/bin/bash
# FOMO25 Task 1 Inference Example
# 
# This script demonstrates how to use the inference pipeline
# with various configuration options.

set -e

# Configuration
CHECKPOINT="runs/Task001_FOMO1/unet_xl/fomo1_optimized_baseline/version_0/checkpoints/best.ckpt"
INPUT_DIR="./example_data"
OUTPUT_DIR="./example_output"

echo "🧠 FOMO25 Task 1 Inference Example"
echo "=================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Example 1: Basic inference
echo "📋 Example 1: Basic inference"
python3 src/inference/predict_task1.py \
    --checkpoint "$CHECKPOINT" \
    --flair "$INPUT_DIR/flair.nii.gz" \
    --adc "$INPUT_DIR/adc.nii.gz" \
    --dwi_b1000 "$INPUT_DIR/dwi_b1000.nii.gz" \
    --t2s "$INPUT_DIR/t2s.nii.gz" \
    --output "$OUTPUT_DIR/prediction_basic.txt" \
    --verbose

# Example 2: With Test-Time Augmentation
echo "📋 Example 2: With Test-Time Augmentation"
python3 src/inference/predict_task1.py \
    --checkpoint "$CHECKPOINT" \
    --flair "$INPUT_DIR/flair.nii.gz" \
    --adc "$INPUT_DIR/adc.nii.gz" \
    --dwi_b1000 "$INPUT_DIR/dwi_b1000.nii.gz" \
    --swi "$INPUT_DIR/swi.nii.gz" \
    --output "$OUTPUT_DIR/prediction_tta.txt" \
    --tta_enable \
    --tta_views 4 \
    --verbose

# Example 3: With ensemble (if multiple models available)
echo "📋 Example 3: With ensemble inference"
python3 src/inference/predict_task1.py \
    --checkpoint "$CHECKPOINT" \
    --ensemble_checkpoints "runs/Task001_FOMO1/unet_xl/ablation_dwi_adc_chanatt/version_0/checkpoints/best.ckpt" \
    --flair "$INPUT_DIR/flair.nii.gz" \
    --adc "$INPUT_DIR/adc.nii.gz" \
    --dwi_b1000 "$INPUT_DIR/dwi_b1000.nii.gz" \
    --t2s "$INPUT_DIR/t2s.nii.gz" \
    --output "$OUTPUT_DIR/prediction_ensemble.txt" \
    --confidence_threshold 0.1 \
    --verbose

# Example 4: Container usage
echo "📋 Example 4: Container build and usage"
echo "# Build container (after updating checkpoint path in apptainer_template.def)"
echo "apptainer build fomo25_task1.sif src/inference/apptainer_template.def"
echo ""
echo "# Run inference in container"
echo "apptainer run \\"
echo "  --bind $INPUT_DIR:/input:ro \\"
echo "  --bind $OUTPUT_DIR:/output:rw \\"
echo "  fomo25_task1.sif \\"
echo "  --flair /input/flair.nii.gz \\"
echo "  --adc /input/adc.nii.gz \\"
echo "  --dwi_b1000 /input/dwi_b1000.nii.gz \\"
echo "  --t2s /input/t2s.nii.gz \\"
echo "  --output /output/prediction_container.txt"

echo ""
echo "✅ Examples completed. Check output files in $OUTPUT_DIR/"
echo ""
echo "📝 Notes:"
echo "  - Make sure to update checkpoint paths in apptainer_template.def"
echo "  - Provide actual input NIfTI files in $INPUT_DIR"
echo "  - Install dependencies: pip install -r src/inference/container_requirements.txt"
