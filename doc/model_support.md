# Supported Model Architectures

## Overview

The FOMO25 inference pipeline supports multiple model architectures with automatic detection and configuration. This document details the supported architectures and their specific requirements.

## Architecture Categories

### 1. Network Architectures

#### UNet Variants
- **unet_xl**: Extended UNet with larger feature channels (64 → 1024)
- **unet_b**: Base UNet architecture (32 → 512 channels)
- **unet_l**: Large UNet variant (48 → 768 channels)

#### Medical-Specific Networks
- **mednext**: Medical ConvNext variant optimized for medical imaging
- **mednext_b**: Base MedNeXt configuration
- **mednext_l**: Large MedNeXt configuration

#### Custom Architectures
- Any architecture defined in `src/models/networks/`
- Automatic parameter detection from checkpoint
- Dynamic model instantiation

### 2. Encoder Configurations

#### Single Encoder (Stacked)
```python
# Traditional approach: concatenated modalities
Input: [B, M, D, H, W]  # M modalities stacked
Encoder: Single network processes all modalities together
Output: Classification/regression head
```

**Detection Logic:**
- Checkpoint contains `model.encoder.in_conv` (not `model.encoder.encoders`)
- Used with stacked preprocessing pipeline
- Legacy compatibility with older training

#### Multi-Encoder Fusion
```python
# Modern approach: per-modality encoders + fusion
Input: [B, M, D, H, W]  # M modalities
Encoders: Separate encoder per modality group
Fusion: Multi-scale feature fusion layers
Output: Classification/regression head
```

**Detection Logic:**
- Checkpoint contains `model.encoder.encoders.{modality_name}`
- Fusion layers: `model.encoder.fusions`
- Automatic modality mapping from checkpoint keys

### 3. Fusion Mechanisms

#### Masked Mean Fusion
```python
class MaskedMeanFusion3D:
    # Weighted average with missing modality handling
    # Learnable gamma parameters per global modality
    # Post-processing with scale and bias
```

**Use Cases:**
- Default robust fusion method
- Handles missing modalities gracefully
- Minimal computational overhead

#### Attention-Based Fusion
```python
class AttentionFusion3D:
    # Cross-modality attention mechanisms
    # Channel, spatial, or hybrid attention
    # Learnable attention weights
```

**Variants:**
- `channel_attention`: Attention across modality channels
- `spatial_attention`: Spatial attention maps
- `hybrid_attention`: Combined channel + spatial

#### Lightweight Fusion
```python
# Efficient fusion mechanisms
fusion_types = [
    "learnable_weighted",    # Learnable per-modality weights
    "channel_gated",         # Channel-wise gating
    "uncertainty_weighted"   # Uncertainty-based weighting
]
```

## Automatic Architecture Detection

### Checkpoint Analysis
```python
def detect_model_architecture(checkpoint_path):
    """
    Analyze checkpoint to determine:
    1. Network architecture (unet_xl, mednext, etc.)
    2. Encoder type (single vs multi-encoder)
    3. Fusion mechanism (if multi-encoder)
    4. Model configuration parameters
    """
    state_dict = torch.load(checkpoint_path)
    
    # Detect encoder type
    if 'model.encoder.encoders' in keys:
        encoder_type = "multi_encoder"
        modalities = extract_modality_names(keys)
        fusion_type = detect_fusion_type(keys)
    else:
        encoder_type = "single_encoder"
        
    # Detect network architecture
    architecture = detect_network_from_layers(keys)
    
    return {
        'architecture': architecture,
        'encoder_type': encoder_type,
        'modalities': modalities,
        'fusion_type': fusion_type,
        'config': extract_config(state_dict)
    }
```

### Configuration Reconstruction
```python
def reconstruct_model_config(checkpoint_info):
    """
    Rebuild config dict compatible with BaseSupervisedModel.create()
    """
    config = {
        'model_name': checkpoint_info['architecture'],
        'use_multi_encoder': checkpoint_info['encoder_type'] == 'multi_encoder',
        'fusion_type': checkpoint_info.get('fusion_type', 'masked_mean'),
        'multi_encoder_modalities': checkpoint_info.get('modalities', []),
        # ... extract other parameters
    }
    return config
```

## Model Loading Pipeline

### 1. Checkpoint Detection
```python
# User-specified checkpoint
if args.checkpoint:
    checkpoint_path = args.checkpoint
# Auto-detect best model
else:
    checkpoint_path = find_best_checkpoint()
```

### 2. Architecture Analysis
```python
model_info = detect_model_architecture(checkpoint_path)
config = reconstruct_model_config(model_info)
```

### 3. Model Instantiation
```python
# Create model using our factory method
model = BaseSupervisedModel.create(
    task_type="classification",
    config=config
)

# Load weights
state_dict = torch.load(checkpoint_path)['state_dict']
model.load_state_dict(state_dict)
model.eval()
```

## Modality Mapping

### Global Modality Groups
```python
GLOBAL_VOCAB = ["t1", "t2", "flair", "dwi", "other"]
```

### Task-Specific Mappings

#### FOMO1 (Task 1)
```python
FOMO1_MAPPING = {
    "DWI": "dwi",           # DWI b1000 images
    "ADC": "dwi",           # ADC maps (dwi group)
    "T2FLAIR": "flair",     # T2 FLAIR images
    "SWI_OR_T2STAR": "other"  # SWI or T2* images
}
```

#### FOMO3 (Task 3)
```python
FOMO3_MAPPING = {
    "T1": "t1",             # T1w images
    "T2": "t2"              # T2w images
}
```

### Inference Argument Mapping
```python
def map_inference_args_to_modalities(args):
    """
    Map command line arguments to canonical modality order
    """
    # Template expects: [dwi_b1000, flair, adc, swi/t2s]
    # Our training uses: ["DWI", "ADC", "T2FLAIR", "SWI_OR_T2STAR"]
    
    modality_paths = [
        args.dwi_b1000,      # Maps to "DWI"
        args.adc,            # Maps to "ADC" 
        args.flair,          # Maps to "T2FLAIR"
        args.swi or args.t2s # Maps to "SWI_OR_T2STAR"
    ]
    
    return modality_paths
```

## Architecture-Specific Considerations

### UNet-XL Models
- **Large memory requirements**: Consider sliding window for large volumes
- **High parameter count**: Ensure sufficient GPU memory
- **Best performance**: Typically highest accuracy

### MedNeXt Models
- **Efficient inference**: Lower memory usage
- **Good generalization**: Robust to domain shifts
- **Faster processing**: Optimized for medical imaging

### Multi-Encoder Fusion
- **Modality flexibility**: Can handle missing modalities
- **Complex preprocessing**: Requires per-modality processing
- **Better feature learning**: Specialized encoders per modality type

### Single Encoder Stacked
- **Simpler pipeline**: Direct modality concatenation
- **Legacy compatibility**: Works with older training
- **Lower complexity**: Fewer parameters and faster inference

## Performance Characteristics

| Architecture | Memory (GB) | Speed (s/case) | Accuracy | Robustness |
|-------------|-------------|----------------|----------|------------|
| unet_xl + fusion | 8-12 | 3-5 | ★★★★★ | ★★★★★ |
| unet_b + fusion | 4-6 | 2-3 | ★★★★ | ★★★★ |
| mednext_l | 3-5 | 2-4 | ★★★★ | ★★★★★ |
| unet_xl stacked | 6-8 | 2-3 | ★★★ | ★★★ |

## Adding New Architectures

### 1. Define Network
```python
# In src/models/networks/new_architecture.py
def new_model(**kwargs):
    return CustomModel(**kwargs)
```

### 2. Register Architecture
```python
# Architecture will be auto-detected from checkpoint
# No explicit registration required
```

### 3. Add Detection Logic
```python
# In inference/model_loader.py
def detect_custom_architecture(state_dict_keys):
    if 'custom_layer' in keys:
        return 'custom_model'
```

### 4. Test Integration
```python
# Verify checkpoint loading
# Test inference pipeline
# Validate output format
```

## Troubleshooting

### Architecture Detection Issues
- **Missing keys**: Checkpoint may be corrupted or incomplete
- **Unknown architecture**: Add detection logic for custom models
- **Config mismatch**: Verify training config compatibility

### Model Loading Failures
- **Memory issues**: Reduce model size or use CPU inference
- **CUDA compatibility**: Check PyTorch/CUDA versions
- **State dict mismatch**: Verify checkpoint format

### Performance Issues
- **Slow inference**: Consider smaller architecture or optimizations
- **High memory usage**: Enable sliding window or reduce batch size
- **Poor accuracy**: Verify preprocessing matches training pipeline
