# Paths
REPO_DIR="/home/weidongguo/workspace/fomo2025/baseline-codebase-main"
SOURCE_DIR="/data/weidong/fomo_finetune/fomo-task3"
OUTPUT_DIR="/data/weidong/fomo-finetune"
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


