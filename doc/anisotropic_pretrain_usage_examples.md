# Anisotropic Pretraining Usage Examples

## Overview
The pretrain script now supports anisotropic patch sizes and anisotropic-aware augmentations, bringing it in line with the advanced capabilities already available in the finetune tasks.

## Basic Usage Examples

### 1. Anisotropic Pretraining with Manual Patch Size
```bash
# Train with anisotropic patches optimized for thick-slice modalities like DWI
python src/pretrain.py \
    --save_dir=/data/weidong/models \
    --pretrain_data_dir=/home/weidongguo/workspace/fomo2025/data \
    --patch_size="96,96,24" \
    --augmentation_preset=anisotropic \
    --modality_mode=dwi \
    --model_name=unet_xl_lw_dec \
    --batch_size=8 \
    --epochs=100
```

### 2. FLAIR Modality with Conservative Anisotropy
```bash
# Train FLAIR modality with moderate anisotropy 
python src/pretrain.py \
    --save_dir=/data/weidong/models \
    --pretrain_data_dir=/home/weidongguo/workspace/fomo2025/data \
    --patch_size="128,128,32" \
    --augmentation_preset=anisotropic \
    --modality_mode=flair \
    --model_name=unet_xl_lw_dec \
    --batch_size=4 \
    --epochs=100
```

### 3. Mixed Modality with Balanced Anisotropy
```bash
# Train all modalities with a balanced anisotropic configuration
python src/pretrain.py \
    --save_dir=/data/weidong/models \
    --pretrain_data_dir=/home/weidongguo/workspace/fomo2025/data \
    --patch_size="96,96,32" \
    --augmentation_preset=anisotropic \
    --modality_mode=all \
    --model_name=unet_xl_lw_dec \
    --batch_size=6 \
    --epochs=100
```

## Backwards Compatibility Examples

### 4. Legacy Isotropic Training (Unchanged)
```bash
# Existing scripts continue to work exactly as before
python src/pretrain.py \
    --save_dir=/data/weidong/models \
    --pretrain_data_dir=/home/weidongguo/workspace/fomo2025/data \
    --patch_size=128 \
    --augmentation_preset=all \
    --modality_mode=t2 \
    --model_name=unet_b_lw_dec \
    --batch_size=4 \
    --epochs=100
```

### 5. String-based Isotropic (New Format)
```bash
# Isotropic training using new string format
python src/pretrain.py \
    --save_dir=/data/weidong/models \
    --pretrain_data_dir=/home/weidongguo/workspace/fomo2025/data \
    --patch_size="128" \
    --augmentation_preset=spatial \
    --modality_mode=t1 \
    --model_name=unet_xl_lw_dec \
    --batch_size=8 \
    --epochs=100
```

## Augmentation Preset Options

### Anisotropic Preset
- **When to use**: For anisotropic patches (ratio > 2:1)
- **Features**: Conservative parameters, reduced through-plane transforms
- **Example**: `--augmentation_preset=anisotropic`

### Standard Presets (Unchanged)
- **none**: No augmentations (fastest)
- **spatial**: Basic spatial augmentations only
- **all**: Full augmentation suite

## Recommended Configurations by Modality

### DWI and ADC (Highly Anisotropic)
```bash
--patch_size="96,96,24" \
--augmentation_preset=anisotropic \
--modality_mode=dwi
```

### FLAIR (Moderately Anisotropic)
```bash
--patch_size="128,128,32" \
--augmentation_preset=anisotropic \
--modality_mode=flair
```

### T1 and T2 (Near Isotropic)
```bash
--patch_size="96" \
--augmentation_preset=spatial \
--modality_mode=t1
```

### Mixed Modalities (Conservative)
```bash
--patch_size="96,96,32" \
--augmentation_preset=anisotropic \
--modality_mode=all
```

## Validation and Debugging

### Check Configuration
The pretrain script now provides detailed logging about anisotropic configuration:

```
[STARTUP] Experiment: dwi_experiment | Modality: dwi | Patch: (96, 96, 24) | Batch: 8 | Devices: 1 | Workers: 4
[ANISOTROPIC] Detected anisotropic patch size with ratio: 4.0
[ANISOTROPIC] Using anisotropic-aware processing for modality: dwi
[ANISOTROPIC] Anisotropic augmentations enabled
```

### Common Validation Errors and Solutions

#### Error: Dimension not divisible by 8
```
AssertionError: Patch size dimension 0 must be divisible by 8, got 95
```
**Solution**: Use patch sizes divisible by 8: `"96,96,24"` instead of `"95,95,23"`

#### Error: Dimension not divisible by mask_patch_size
```
AssertionError: Patch size dimension 2 must be divisible by mask_patch_size 8, got 25
```
**Solution**: Ensure all dimensions are divisible by `--mask_patch_size`: use `"96,96,24"` with `--mask_patch_size=4`

#### Error: mask_patch_size too large
```
AssertionError: mask_patch_size 16 must be smaller than minimum patch dimension 8
```
**Solution**: Use smaller `--mask_patch_size` or larger patch dimensions

## Performance Notes

### Memory Usage
- Anisotropic patches can be more memory-efficient than equivalent isotropic volumes
- `(96,96,24)` uses ~55% less memory than `(96,96,96)` 
- Allows larger batch sizes or more complex models

### Training Speed
- Anisotropic patches typically train faster due to smaller volume
- Conservative augmentations may be slightly slower due to anisotropy checking
- Overall performance usually improved due to better memory utilization

### Convergence
- Anisotropic pretraining often converges faster
- Better feature learning for anisotropic downstream tasks
- Reduced interpolation artifacts compared to forced isotropic resampling

## Migration Guide

### From Existing Pretrain Scripts
1. **No changes needed**: Existing scripts work unchanged
2. **Optional upgrade**: Add anisotropic support for better performance
3. **Gradual migration**: Test anisotropic on single modalities first

### Example Migration
```bash
# BEFORE (existing script)
python src/pretrain.py --patch_size=128 --augmentation_preset=all

# AFTER (optimized anisotropic)  
python src/pretrain.py --patch_size="128,128,32" --augmentation_preset=anisotropic
```

## Testing Your Configuration

Use our test script to validate your anisotropic configuration:
```bash
python3 test_anisotropic_pretrain.py
```

This will verify:
- ✅ Patch size parsing works correctly
- ✅ Anisotropy detection functions properly  
- ✅ Validation logic catches invalid inputs
- ✅ Augmentation functions can be imported (requires PyTorch environment)

## Summary

The anisotropic pretraining support brings several advantages:
- 🎯 **Better data matching**: Patches match actual medical imaging characteristics
- 🚀 **Improved efficiency**: More efficient memory and compute utilization
- 🔗 **Seamless transfer**: Better initialization for anisotropic downstream tasks
- 🔄 **Backwards compatible**: Existing workflows continue unchanged
- 📊 **Proven approach**: Based on successful finetune implementation

Start with the recommended configurations above and adjust based on your specific data characteristics and computational resources.
