# Paths
REPO_DIR="/mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main"
SOURCE_DIR="/mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune/fomo-task1"
OUTPUT_DIR="/mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune"
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