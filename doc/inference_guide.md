# FOMO25 Inference Guide

## Overview

This guide provides comprehensive documentation for the FOMO25 inference pipeline, supporting flexible model architectures, user-configurable checkpoints, and advanced inference aggregation strategies.

## Quick Start

### Basic Usage
```bash
python src/inference/predict_task1.py \
  --flair /path/to/flair.nii.gz \
  --adc /path/to/adc.nii.gz \
  --dwi_b1000 /path/to/dwi_b1000.nii.gz \
  --t2s /path/to/t2s.nii.gz \
  --output /path/to/output.txt
```

### With Custom Checkpoint
```bash
python src/inference/predict_task1.py \
  --checkpoint runs/Task001_FOMO1/unet_xl/custom_model/version_0/checkpoints/best.ckpt \
  --flair /path/to/flair.nii.gz \
  --adc /path/to/adc.nii.gz \
  --dwi_b1000 /path/to/dwi_b1000.nii.gz \
  --swi /path/to/swi.nii.gz \
  --output /path/to/output.txt
```

### With Inference Aggregation
```bash
python src/inference/predict_task1.py \
  --checkpoint runs/Task001_FOMO1/unet_xl/model1/version_0/checkpoints/best.ckpt \
  --ensemble_checkpoints runs/Task001_FOMO1/unet_xl/model2/version_0/checkpoints/best.ckpt,runs/Task001_FOMO1/unet_xl/model3/version_0/checkpoints/best.ckpt \
  --tta_enable \
  --tta_views 8 \
  --confidence_threshold 0.1 \
  --flair /path/to/flair.nii.gz \
  --adc /path/to/adc.nii.gz \
  --dwi_b1000 /path/to/dwi_b1000.nii.gz \
  --t2s /path/to/t2s.nii.gz \
  --output /path/to/output.txt
```

## Command Line Arguments

### Required Arguments
- `--flair`: Path to T2 FLAIR image (NIfTI format)
- `--adc`: Path to ADC image (NIfTI format) 
- `--dwi_b1000`: Path to DWI b1000 image (NIfTI format)
- `--output`: Path to save output .txt file with probability

### Modality Arguments (One Required)
- `--t2s`: Path to T2* image (NIfTI format)
- `--swi`: Path to SWI image (NIfTI format)

### Model Configuration
- `--checkpoint`: Path to specific checkpoint (auto-detects best if not specified)
- `--model_architecture`: Force specific architecture (unet_xl, etc.)
- `--fusion_type`: Override fusion mechanism (masked_mean, attention, etc.)

### Inference Aggregation
- `--ensemble_checkpoints`: Comma-separated list of additional checkpoints for ensemble
- `--tta_enable`: Enable Test-Time Augmentation
- `--tta_views`: Number of TTA views (default: 8)
- `--tta_offsets`: Number of spatial offsets (default: 1)
- `--sliding_window`: Enable sliding window for large volumes
- `--confidence_threshold`: Minimum confidence for prediction

### Performance Options
- `--batch_size`: Inference batch size (default: 1)
- `--num_workers`: Number of preprocessing workers (default: 4)
- `--device`: Force specific device (cuda:0, cpu, etc.)

## Architecture Support

The inference pipeline automatically detects and supports:

### Model Architectures
- **UNet-XL**: Large UNet with extended feature channels
- **UNet-B**: Base UNet architecture
- **MedNeXt**: Medical-specific ConvNext variant
- Custom architectures from `src/models/networks/`

### Fusion Mechanisms
- **Masked Mean**: Weighted average with missing modality handling
- **Attention Fusion**: Cross-modality attention mechanisms
- **Channel/Spatial/Hybrid Attention**: Specialized attention variants
- **Lightweight Fusion**: Learnable weighted, channel gated, uncertainty weighted

### Model Types
- **Multi-Encoder Fusion**: Per-modality encoders with fusion layers
- **Single Encoder Stacked**: Traditional stacked modality input

## Preprocessing Pipeline

The inference preprocessing automatically:

1. **Detects canonical modalities** using filename patterns
2. **Loads and validates** NIfTI images
3. **Applies joint preprocessing** for spatial alignment
4. **Normalizes intensities** using volume-wise z-score
5. **Crops to non-zero** regions to remove empty space
6. **Handles missing modalities** with zero-padding
7. **Converts to tensor format** for model input

## Inference Aggregation Methods

### Test-Time Augmentation (TTA)
- **Flips**: Up to 8 deterministic flip combinations
- **Spatial Offsets**: Multiple patch positions for large volumes
- **Rotation**: Optional rotation augmentations
- **Aggregation**: Average predictions across augmentations

### Ensemble Methods
- **Multi-Model**: Average predictions from different trained models
- **Multi-Architecture**: Combine different network architectures
- **Multi-Fusion**: Ensemble different fusion mechanisms
- **Weighted Averaging**: Performance-based model weighting

### Sliding Window
- **Large Volume Support**: Handle volumes exceeding patch size
- **Overlap Strategy**: Configurable overlap for smooth predictions
- **Memory Management**: Efficient processing of large datasets

### Uncertainty Estimation
- **Prediction Confidence**: Softmax entropy-based confidence
- **Model Agreement**: Ensemble disagreement metrics
- **Calibrated Probabilities**: Temperature scaling for better calibration

## Container Usage

### Build Container
```bash
# Update checkpoint path in apptainer_template.def
apptainer build fomo25_task1.sif src/inference/apptainer_template.def
```

### Run Inference in Container
```bash
apptainer run \
  --bind /path/to/input:/input:ro \
  --bind /path/to/output:/output:rw \
  fomo25_task1.sif \
  --flair /input/flair.nii.gz \
  --adc /input/adc.nii.gz \
  --dwi_b1000 /input/dwi_b1000.nii.gz \
  --t2s /input/t2s.nii.gz \
  --output /output/prediction.txt
```

## Output Format

The inference pipeline outputs a single probability value indicating the likelihood of infarct presence:

```
0.847
```

Where:
- Values range from 0.0 to 1.0
- Higher values indicate higher probability of infarct
- Values > 0.5 typically indicate positive classification
- Confidence thresholding available for uncertain predictions

## Troubleshooting

### Common Issues

**Model Loading Errors**
- Verify checkpoint path exists
- Check CUDA/CPU compatibility
- Ensure model architecture matches training

**Preprocessing Failures**
- Validate NIfTI file integrity
- Check modality detection patterns
- Verify spatial alignment

**Memory Issues**
- Reduce batch size
- Enable sliding window for large volumes
- Use CPU inference for very large cases

**Container Issues**
- Check Apptainer installation
- Verify mount paths
- Ensure checkpoint is copied to container

## Performance Optimization

### Speed Optimization
- Use GPU inference when available
- Optimize batch size for your hardware
- Disable unnecessary aggregation methods
- Use compiled models for production

### Accuracy Optimization
- Enable Test-Time Augmentation
- Use ensemble methods with diverse models
- Apply confidence thresholding for uncertain cases
- Use calibrated probabilities for better reliability

## Development Notes

This inference pipeline is designed for:
- **Production deployment** in containerized environments
- **Research flexibility** with multiple model architectures
- **Robust handling** of real-world data variability
- **Performance optimization** for clinical workflows

For implementation details, see the source code in `src/inference/` and additional documentation in `doc/`.
