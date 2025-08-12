#!/usr/bin/env bash
set -euo pipefail

# Paths
REPO_DIR="/mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main"
SOURCE_DIR="/mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune/fomo-task3"
OUTPUT_DIR="/mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune"
NUM_WORKERS="${NUM_WORKERS:-$(nproc)}"

# Activate env
cd "${REPO_DIR}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate fomo || true
fi

# Run preprocessing (fusion)
PYTHONPATH=src python "${REPO_DIR}/src/data/preprocess/fomo3_fusion.py" \
  --source_path "${SOURCE_DIR}" \
  --output_path "${OUTPUT_DIR}" \
  --num_workers "${NUM_WORKERS}"

echo "Done. Check: ${OUTPUT_DIR}/Task003_FOMO3_fusion"


