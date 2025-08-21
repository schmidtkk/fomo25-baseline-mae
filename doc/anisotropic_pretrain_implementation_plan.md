# Anisotropic Support Implementation Plan for Pretraining

## Executive Summary

This document outlines the comprehensive plan to add anisotropic patch size and data augmentation support to the FOMO 2025 pretraining process. The current codebase has excellent anisotropic support in the finetune tasks, but the pretrain process is limited to isotropic (cubic) patches.

## Current State Analysis

### ✅ What's Already Working (Finetune)
- **Anisotropic patch size parsing**: `patch_size="96,96,24"` → `(96, 96, 24)`
- **Task-specific configurations**: Each task has optimized anisotropic patch sizes
- **Advanced augmentations**: `AnisotropicSpatialTransform` with axis-specific parameters
- **Integration**: Seamless integration with YuccaDataModule and YuccaAugmentationComposer

### ❌ What's Missing (Pretrain)
- **Single integer patch size**: Currently only supports `--patch_size=128` (isotropic)
- **Basic augmentations**: Only uses standard spatial transforms, no anisotropic awareness
- **No task-specific pretraining**: No differentiation between modalities' anisotropy characteristics
- **Masking assumptions**: Assumes cubic patches for MAE masking

## Implementation Plan

### Phase 1: Analysis and Documentation ✅ COMPLETED
**Goal**: Understand the current implementation and document findings  
**Status**: ✅ COMPLETE  
**Key Findings**:
- Finetune uses sophisticated `AnisotropicSpatialTransform` class
- Pretrain uses basic `Spatial` transform from yucca
- Patch size parsing in finetune supports both formats
- MAE masking works with any patch shape

### Phase 2: Anisotropic Patch Size Support 
**Goal**: Add anisotropic patch size support to pretrain.py  
**Status**: 🔄 IN PROGRESS  

#### 2.1 Argument Parsing Enhancement
**File**: `src/pretrain.py`
```python
# BEFORE (line ~53)
parser.add_argument(
    "--patch_size",
    type=int,
    default=64,
    help="The patch size of the 3D patches extracted from the whole volume.",
)

# AFTER
parser.add_argument(
    "--patch_size", 
    type=str,
    default="64",
    help="Patch size: single int for isotropic (64) or comma-separated for anisotropic (96,96,24)",
)
```

#### 2.2 Patch Size Processing Logic
**Location**: After argument parsing in `main()`
```python
# Parse patch size: support both isotropic (int) and anisotropic (comma-separated)
if isinstance(args.patch_size, str):
    if ',' in args.patch_size:
        # Anisotropic: "96,96,24" -> (96, 96, 24)  
        patch_size_tuple = tuple(int(x.strip()) for x in args.patch_size.split(','))
        assert len(patch_size_tuple) == 3, f"Anisotropic patch size must have 3 dimensions, got {len(patch_size_tuple)}"
    else:
        # Isotropic: "96" -> (96, 96, 96)
        patch_size_int = int(args.patch_size)
        patch_size_tuple = (patch_size_int,) * 3
else:
    # Legacy int support
    patch_size_tuple = (args.patch_size,) * 3

# Validate all dimensions are divisible by 8 AND mask_patch_size
for i, dim in enumerate(patch_size_tuple):
    assert dim % 8 == 0, f"Patch size dimension {i} must be divisible by 8, got {dim}"
    assert dim % args.mask_patch_size == 0, f"Patch size dimension {i} must be divisible by mask_patch_size {args.mask_patch_size}, got {dim}"
```

#### 2.3 Configuration Update
**Location**: Update config dictionary
```python
# BEFORE
"patch_size": (args.patch_size,) * 3,

# AFTER  
"patch_size": patch_size_tuple,
```

### Phase 3: Anisotropic Augmentation Support
**Goal**: Add anisotropic-aware augmentations to pretraining  
**Status**: 📋 PLANNED

#### 3.1 Augmentation Preset Extension
**File**: `src/augmentations/augmentation_composer.py`

Add new preset option:
```python
def get_pretrain_augmentations(patch_size, preset):
    assert preset in ["none", "spatial", "all", "anisotropic"]  # Add anisotropic

    if preset == "none":
        augmentations = [CopyImageToLabel(copy=True)]
    elif preset == "spatial":
        augmentations = [spatial_augmentation(patch_size), CopyImageToLabel(copy=True)]
    elif preset == "anisotropic":
        augmentations = [anisotropic_spatial_augmentation(patch_size), CopyImageToLabel(copy=True)]
    elif preset == "all":
        augmentations = [
            spatial_augmentation(patch_size),
            CopyImageToLabel(copy=True),
        ] + intensity_augmentations()
```

#### 3.2 Anisotropic Transform Function
**File**: `src/augmentations/augmentation_composer.py`

Add new function:
```python
def anisotropic_spatial_augmentation(patch_size):
    """
    Create anisotropic-aware spatial augmentation for pretraining.
    
    Uses conservative parameters suitable for mixed-modality pretraining data.
    """
    # Calculate anisotropy ratio to determine if special handling is needed
    max_dim = max(patch_size)
    min_dim = min(patch_size)
    anisotropy_ratio = max_dim / min_dim if min_dim > 0 else 1.0
    
    if anisotropy_ratio > 2.0:  # Anisotropic data detected
        return Spatial(
            patch_size=patch_size,
            crop=True,
            random_crop=False,
            cval="min",
            # Conservative deformation for thick-slice data
            p_deform_per_sample=0.2,  # Reduced from 0.33
            deform_sigma=(15, 25),    # Reduced from (20, 30)
            deform_alpha=(150, 400),  # Reduced from (200, 600)
            # Axis-aware rotation (reduced for through-plane)
            p_rot_per_sample=0.15,    # Reduced from 0.2
            p_rot_per_axis=0.5,       # Reduced from 0.66
            x_rot_in_degrees=(-15.0, 15.0),  # Reduced from (-30, 30)
            y_rot_in_degrees=(-15.0, 15.0),  # Reduced from (-30, 30) 
            z_rot_in_degrees=(-20.0, 20.0),  # Slightly reduced
            # Conservative scaling for through-plane axis
            p_scale_per_sample=0.15,  # Reduced from 0.2
            scale_factor=(0.95, 1.05), # More conservative than (0.9, 1.1)
            skip_label=True,
            clip_to_input_range=True,
        )
    else:
        # Use standard parameters for isotropic data
        return spatial_augmentation(patch_size)
```

#### 3.3 Pretrain Argument Extension  
**File**: `src/pretrain.py`

Update argument choices:
```python
parser.add_argument(
    "--augmentation_preset",
    type=str,
    choices=["all", "basic", "none", "anisotropic"],  # Add anisotropic
    default="none",
)
```

### Phase 4: Modality-Aware Pretraining Configuration
**Goal**: Enable modality-specific anisotropic pretraining  
**Status**: 📋 PLANNED

#### 4.1 Modality-Specific Patch Size Defaults
**File**: `src/pretrain.py`

Add modality-aware defaults:
```python
# Add after argument parsing
def get_modality_default_patch_size(modality_mode: str) -> tuple:
    """Get default anisotropic patch sizes based on modality characteristics."""
    modality_defaults = {
        "t1": (96, 96, 96),      # T1 is typically isotropic
        "t2": (96, 96, 96),      # T2 is typically isotropic  
        "flair": (96, 96, 32),   # FLAIR often thicker slices
        "dwi": (96, 96, 24),     # DWI typically anisotropic
        "other": (96, 96, 24),   # Other modalities often anisotropic
        "all": (96, 96, 32),     # Conservative for mixed data
    }
    return modality_defaults.get(modality_mode.lower(), (96, 96, 96))

# Use in main() if patch_size is default
if args.patch_size == "64":  # Default value
    modality_patch_size = get_modality_default_patch_size(args.modality_mode)
    patch_size_tuple = modality_patch_size
    print(f"📐 Using modality-specific anisotropic patch size for {args.modality_mode}: {patch_size_tuple}")
else:
    # User specified, parse normally
    # ... existing parsing logic
```

#### 4.2 Logging Enhancement
Add better logging for anisotropic configurations:
```python
logging.info(f"Patch size configuration: {patch_size_tuple}")
if patch_size_tuple[0] != patch_size_tuple[1] or patch_size_tuple[1] != patch_size_tuple[2]:
    anisotropy_ratio = max(patch_size_tuple) / min(patch_size_tuple)
    logging.info(f"Detected anisotropic patch size with ratio: {anisotropy_ratio:.2f}")
    logging.info(f"Using anisotropic-aware augmentations for modality: {args.modality_mode}")
else:
    logging.info("Using isotropic patch size configuration")
```

### Phase 5: Validation and Testing
**Goal**: Ensure anisotropic pretraining works correctly  
**Status**: 📋 PLANNED

#### 5.1 MAE Masking Compatibility
**Test**: Verify MAE masking works with anisotropic patches
```python
# Test in src/utils/masking.py
def test_anisotropic_masking():
    # Test cases
    test_cases = [
        ((96, 96, 24), 4),  # Anisotropic with mask_patch_size=4
        ((128, 128, 32), 8), # Anisotropic with mask_patch_size=8
        ((64, 64, 64), 4),   # Isotropic control
    ]
    
    for patch_size, mask_patch_size in test_cases:
        x = torch.randn(2, 1, *patch_size)  # Batch of 2, 1 channel
        mask = generate_random_mask(x, mask_ratio=0.6, patch_size=mask_patch_size)
        assert mask.shape == x.shape, f"Mask shape {mask.shape} != data shape {x.shape}"
        print(f"✅ Masking works for patch_size={patch_size}, mask_patch_size={mask_patch_size}")
```

#### 5.2 Data Pipeline Testing  
**Test**: Verify data loading and augmentation with anisotropic patches
```python
# Test script
def test_anisotropic_pretrain_pipeline():
    from src.data.datamodule import PretrainDataModule
    from src.augmentations.augmentation_composer import get_pretrain_augmentations
    
    patch_size = (96, 96, 24)
    transforms = get_pretrain_augmentations(patch_size, "anisotropic")
    
    # Create dummy data module (would need actual data path)
    data_module = PretrainDataModule(
        patch_size=patch_size,
        batch_size=2,
        num_workers=1,
        # ... other params
    )
    
    # Test data loading
    data_module.setup("fit") 
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"✅ Anisotropic pretraining pipeline works:")
    print(f"   Batch shape: {batch['image'].shape}")
    print(f"   Expected: (2, 1, 96, 96, 24)")
```

## Implementation Schedule

### Week 1: Core Patch Size Support
- [x] ~~Document current state and plan~~
- [ ] Implement patch size parsing in pretrain.py
- [ ] Update configuration and validation
- [ ] Test with simple anisotropic patches

### Week 2: Augmentation Integration  
- [ ] Add anisotropic preset to augmentation_composer.py
- [ ] Implement anisotropic_spatial_augmentation function
- [ ] Test augmentation pipeline with anisotropic data
- [ ] Validate MAE masking compatibility

### Week 3: Modality-Aware Features
- [ ] Add modality-specific patch size defaults  
- [ ] Enhanced logging and configuration
- [ ] Integration testing with different modalities
- [ ] Performance benchmarking

### Week 4: Final Testing and Documentation
- [ ] Comprehensive end-to-end testing
- [ ] Performance comparison: isotropic vs anisotropic
- [ ] Documentation updates
- [ ] Script updates and examples

## Expected Benefits

### 1. Improved Pretraining Quality
- **Better Feature Learning**: Anisotropic patches match the actual data characteristics
- **Reduced Artifacts**: Avoid interpolation artifacts from forced isotropic resampling
- **Modality-Specific Optimization**: Each modality gets appropriate patch dimensions

### 2. Enhanced Transfer Learning  
- **Better Initialization**: Pretrained weights will better match finetuning patch sizes
- **Improved Convergence**: Finetuning should converge faster with matching geometries
- **Task-Specific Benefits**: Each downstream task benefits from appropriate pretraining

### 3. Resource Efficiency
- **Memory Optimization**: Anisotropic patches can be more memory-efficient
- **Compute Efficiency**: Process actual data dimensions rather than artificial cubic volumes
- **Storage Benefits**: Better utilization of available voxel information

## Risk Mitigation

### Backwards Compatibility
- **Legacy Support**: Keep existing `--patch_size=128` format working
- **Default Behavior**: Default to isotropic if no anisotropy specified
- **Gradual Migration**: Allow users to migrate incrementally

### Performance Validation  
- **Benchmark Testing**: Compare pretraining convergence isotropic vs anisotropic
- **Downstream Validation**: Test impact on finetuning performance
- **Memory Profiling**: Ensure memory usage remains reasonable

### Error Handling
- **Input Validation**: Comprehensive validation of patch size inputs
- **Clear Error Messages**: Helpful error messages for invalid configurations  
- **Fallback Behavior**: Graceful fallback to isotropic when needed

## Success Criteria

### Primary Goals ✅
1. **Patch Size Support**: `--patch_size="96,96,24"` works in pretrain.py
2. **Augmentation Integration**: `--augmentation_preset=anisotropic` available
3. **Backwards Compatibility**: Existing scripts continue to work unchanged
4. **MAE Compatibility**: Masking works correctly with anisotropic patches

### Secondary Goals 📋
1. **Modality-Aware Defaults**: Automatic patch size selection per modality
2. **Performance Improvement**: Measurable improvement in downstream tasks
3. **Documentation**: Comprehensive usage guides and examples
4. **Testing**: Full test coverage for anisotropic functionality

## Usage Examples

### Basic Anisotropic Pretraining
```bash
# Example 1: Manual anisotropic patch size
python src/pretrain.py \
    --save_dir=/data/models \
    --pretrain_data_dir=/data/FOMO60k \
    --patch_size="96,96,24" \
    --augmentation_preset=anisotropic \
    --modality_mode=dwi

# Example 2: Modality-aware automatic sizing (future)
python src/pretrain.py \
    --save_dir=/data/models \
    --pretrain_data_dir=/data/FOMO60k \
    --augmentation_preset=anisotropic \
    --modality_mode=flair  # Will auto-select appropriate anisotropic patch size
```

### Backwards Compatible Usage
```bash
# Existing scripts continue to work unchanged
python src/pretrain.py \
    --save_dir=/data/models \
    --pretrain_data_dir=/data/FOMO60k \
    --patch_size=128 \
    --augmentation_preset=all
```

## Conclusion

This implementation plan provides a comprehensive roadmap for adding anisotropic support to the FOMO 2025 pretraining process. The approach prioritizes backwards compatibility while adding powerful new anisotropic capabilities that match the sophisticated support already available in the finetuning tasks.

The phased implementation allows for incremental development and testing, with clear success criteria and risk mitigation strategies. Once complete, the pretraining process will be significantly more aligned with the characteristics of medical imaging data and should provide better initialization for downstream tasks.

---

**Status**: 📋 Ready for implementation  
**Next Step**: Begin Phase 2 implementation with patch size support  
**Review Required**: ✅ Plan approved for execution
