#!/bin/bash
# FOMO Task 2 Preprocessing Script - Meningioma Segmentation
# Task 2: Binary segmentation of brain meningiomas on MRI scans
# Modalities: T2 FLAIR, DWI (b-value 1000), and either T2* or SWI images

# Paths
REPO_DIR="/home/weidongguo/workspace/fomo2025/baseline-codebase-main"
SOURCE_DIR="/data/weidong/fomo_finetune/fomo-task2"
OUTPUT_DIR="/data/weidong/fomo-finetune"
NUM_WORKERS="${NUM_WORKERS:-$(nproc)}"  # override by exporting NUM_WORKERS if desired

echo "🧠 FOMO Task 2 - Meningioma Segmentation Preprocessing"
echo "======================================================"
echo "📋 Task: Binary segmentation of brain meningiomas"
echo "🧬 Modalities: DWI, T2FLAIR, SWI_OR_T2STAR"
echo "📊 Output: Dice & NSD metrics for segmentation evaluation"
echo "📁 Source: ${SOURCE_DIR}"
echo "📁 Output: ${OUTPUT_DIR}/Task002_FOMO2_fusion"
echo ""

# Activate environment
cd "${REPO_DIR}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate fomo || true
fi

# Run preprocessing (fusion)
PYTHONPATH=src python "${REPO_DIR}/src/data/preprocess/fomo2_fusion.py" \
  --source_path "${SOURCE_DIR}" \
  --output_path "${OUTPUT_DIR}" \
  --num_workers "${NUM_WORKERS}"

echo ""
echo "✅ Done! Check output: ${OUTPUT_DIR}/Task002_FOMO2_fusion"
echo ""
echo "📂 Expected structure per subject:"
echo "   <subject>/"
echo "   ├── DWI.npy"
echo "   ├── T2FLAIR.npy" 
echo "   ├── SWI_OR_T2STAR.npy"
echo "   ├── mask.json"
echo "   ├── seg.npy          # for training"
echo "   └── seg.nii.gz       # for evaluation"
