# Paths
REPO_DIR="/home/weidongguo/workspace/fomo2025/baseline-codebase-main"
SOURCE_DIR="/data/weidong/fomo_finetune/fomo-task1"
OUTPUT_DIR="/data/weidong/fomo-finetune"
NUM_WORKERS="${NUM_WORKERS:-$(nproc)}"  # override by exporting NUM_WORKERS if desired

# Activate env
cd "${REPO_DIR}"
eval "$(conda shell.bash hook)"
conda activate fomo

# Run preprocessing (fusion)
PYTHONPATH=src python "${REPO_DIR}/src/data/preprocess/fomo1_fusion.py" \
  --source_path "${SOURCE_DIR}" \
  --output_path "${OUTPUT_DIR}" \
  --num_workers "${NUM_WORKERS}"

echo "Done. Check: ${OUTPUT_DIR}/Task001_FOMO1_fusion"