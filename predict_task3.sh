#!/bin/bash
# FOMO Task 3 - Brain Age Prediction Inference Script

set -e

# Default configuration
DEFAULT_CHECKPOINT="runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/checkpoints/best.ckpt"
DEFAULT_CONFIG="runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/hparams.yaml"
DEFAULT_DEVICE="auto"
DEFAULT_BATCH_SIZE="1"
DEFAULT_OUTPUT_NAME="brain_age_prediction"
TTA_FLAG=""
OUTPUT_DIR=""
MODALITIES=()
VERBOSE_FLAG=""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧠 FOMO Task 3 - Brain Age Prediction${NC}"
echo -e "${BLUE}====================================${NC}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --modalities)
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        MODALITIES+=("$1")
        shift
      done
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --checkpoint_path)
      DEFAULT_CHECKPOINT="$2"
      shift 2
      ;;
    --config_path)
      DEFAULT_CONFIG="$2"
      shift 2
      ;;
    --device)
      DEFAULT_DEVICE="$2"
      shift 2
      ;;
    --batch_size)
      DEFAULT_BATCH_SIZE="$2"
      shift 2
      ;;
    --output_name)
      DEFAULT_OUTPUT_NAME="$2"
      shift 2
      ;;
    --tta)
      TTA_FLAG="--tta"
      shift
      ;;
    --verbose)
      VERBOSE_FLAG="--verbose"
      shift
      ;;
    -h|--help)
      echo -e "${GREEN}FOMO Task 3 - Brain Age Prediction${NC}"
      echo ""
      echo "Usage: $0 --modalities <t1.nii.gz> <t2.nii.gz> --output_dir <dir> [options]"
      echo ""
      echo "Required arguments:"
      echo "  --modalities T1_PATH T2_PATH    Paths to T1 and T2 NIfTI files"
      echo "  --output_dir DIR                Directory to save results"
      echo ""
      echo "Optional arguments:"
      echo "  --checkpoint_path PATH          Custom checkpoint path"
      echo "  --config_path PATH              Custom config path"
      echo "  --device DEVICE                 Device (auto, cpu, cuda, cuda:X)"
      echo "  --batch_size SIZE               Inference batch size (default: 1)"
      echo "  --output_name NAME              Output file base name"
      echo "  --tta                           Enable test-time augmentation"
      echo "  --verbose                       Enable verbose logging"
      echo "  -h, --help                      Show this help message"
      echo ""
      echo "Examples:"
      echo "  # Basic usage"
      echo "  $0 --modalities subject_t1.nii.gz subject_t2.nii.gz --output_dir ./results"
      echo ""
      echo "  # Advanced usage with TTA"
      echo "  $0 --modalities t1.nii.gz t2.nii.gz --output_dir ./results --tta --device cuda:1"
      echo ""
      exit 0
      ;;
    *)
      echo -e "${RED}Error: Unknown argument: $1${NC}"
      echo "Use -h or --help for usage information."
      exit 1
      ;;
  esac
done

# Validation
if [ ${#MODALITIES[@]} -eq 0 ]; then
  echo -e "${RED}Error: --modalities is required${NC}"
  echo "Specify T1 and T2 NIfTI files: --modalities t1.nii.gz t2.nii.gz"
  exit 1
fi

if [ ${#MODALITIES[@]} -ne 2 ]; then
  echo -e "${RED}Error: Exactly 2 modalities required (T1 and T2)${NC}"
  echo "Got: ${MODALITIES[*]}"
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  echo -e "${RED}Error: --output_dir is required${NC}"
  exit 1
fi

# Check if modality files exist
for i in "${!MODALITIES[@]}"; do
  if [ ! -f "${MODALITIES[$i]}" ]; then
    echo -e "${RED}Error: Modality file not found: ${MODALITIES[$i]}${NC}"
    exit 1
  fi
done

# Display configuration
echo -e "${YELLOW}📊 Configuration:${NC}"
echo -e "  T1 Modality: ${MODALITIES[0]}"
echo -e "  T2 Modality: ${MODALITIES[1]}"
echo -e "  Output Directory: $OUTPUT_DIR"
echo -e "  Device: $DEFAULT_DEVICE"
echo -e "  Checkpoint: $DEFAULT_CHECKPOINT"

if [ -n "$TTA_FLAG" ]; then
  echo -e "  ${GREEN}✓${NC} Test-Time Augmentation enabled"
fi

if [ -n "$VERBOSE_FLAG" ]; then
  echo -e "  ${GREEN}✓${NC} Verbose logging enabled"
fi

echo ""

# Check if Python environment is available
if ! command -v /home/weidongguo/miniconda3/envs/fomo/bin/python &> /dev/null; then
  echo -e "${RED}Error: FOMO Python environment not found${NC}"
  echo "Expected: /home/weidongguo/miniconda3/envs/fomo/bin/python"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Execute prediction
echo -e "${BLUE}🚀 Running brain age prediction...${NC}"
echo ""

PYTHONPATH=src /home/weidongguo/miniconda3/envs/fomo/bin/python src/inference/predict_task3.py \
  --modalities "${MODALITIES[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint_path "$DEFAULT_CHECKPOINT" \
  --config_path "$DEFAULT_CONFIG" \
  --device "$DEFAULT_DEVICE" \
  --batch_size "$DEFAULT_BATCH_SIZE" \
  --output_name "$DEFAULT_OUTPUT_NAME" \
  $TTA_FLAG \
  $VERBOSE_FLAG

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
  echo -e "${GREEN}✅ Prediction completed successfully!${NC}"
  echo -e "📁 Results saved in: $OUTPUT_DIR"
else
  echo -e "${RED}❌ Prediction failed with exit code: $exit_code${NC}"
  echo "Check the error messages above for details."
fi

exit $exit_code
