# Inference Aggregation Methods

## Overview

The FOMO25 inference pipeline supports multiple aggregation strategies to improve prediction reliability and accuracy. These methods can be used individually or in combination to achieve robust inference performance.

## Aggregation Categories

### 1. Test-Time Augmentation (TTA)

#### Deterministic TTA
**Geometric Transformations**
```python
tta_transforms = [
    "original",           # No transformation
    "flip_x",            # Flip along X axis
    "flip_y",            # Flip along Y axis  
    "flip_z",            # Flip along Z axis
    "flip_xy",           # Flip along X and Y
    "flip_xz",           # Flip along X and Z
    "flip_yz",           # Flip along Y and Z
    "flip_xyz"           # Flip along all axes
]
```

**Spatial Offsets**
```python
# For volumes larger than patch size
spatial_offsets = [
    "center",            # Standard center crop
    "offset_x+",         # Shift in +X direction
    "offset_x-",         # Shift in -X direction
    "offset_y+",         # Shift in +Y direction
    "offset_y-",         # Shift in -Y direction
    "offset_z+",         # Shift in +Z direction
    "offset_z-",         # Shift in -Z direction
]
```

#### Stochastic TTA
```python
# Random augmentations during inference
stochastic_tta = [
    "noise_injection",    # Gaussian noise addition
    "intensity_shift",    # Random intensity scaling
    "elastic_deform",     # Elastic deformation
    "rotation_small",     # Small random rotations
]
```

#### TTA Implementation
```python
def apply_tta_inference(model, input_tensor, tta_config):
    """
    Apply Test-Time Augmentation for robust prediction
    """
    predictions = []
    
    for transform in tta_config['transforms']:
        # Apply forward transform
        augmented_input = apply_transform(input_tensor, transform)
        
        # Run inference
        with torch.no_grad():
            pred = model(augmented_input)
        
        # Apply inverse transform to prediction
        pred_orig = apply_inverse_transform(pred, transform)
        predictions.append(pred_orig)
    
    # Aggregate predictions
    if tta_config['aggregation'] == 'mean':
        final_pred = torch.mean(torch.stack(predictions), dim=0)
    elif tta_config['aggregation'] == 'median':
        final_pred = torch.median(torch.stack(predictions), dim=0)[0]
    
    return final_pred, predictions
```

### 2. Ensemble Methods

#### Multi-Model Ensemble
```python
def ensemble_multi_model(checkpoints, input_data):
    """
    Ensemble predictions from multiple trained models
    """
    predictions = []
    weights = []
    
    for checkpoint_path in checkpoints:
        model = load_model(checkpoint_path)
        pred = model(input_data)
        predictions.append(pred)
        
        # Performance-based weighting
        weight = get_model_performance(checkpoint_path)
        weights.append(weight)
    
    # Weighted ensemble
    weighted_pred = sum(p * w for p, w in zip(predictions, weights))
    weighted_pred /= sum(weights)
    
    return weighted_pred, predictions
```

#### Multi-Architecture Ensemble
```python
architectures = [
    "unet_xl",           # Large UNet
    "mednext_l",         # Large MedNeXt
    "custom_model"       # Custom architecture
]

def ensemble_multi_architecture(architectures, input_data):
    """
    Combine predictions from different network architectures
    """
    predictions = {}
    
    for arch in architectures:
        model = load_best_model(architecture=arch)
        pred = model(input_data)
        predictions[arch] = pred
    
    # Architecture-specific weighting
    arch_weights = {
        "unet_xl": 0.4,      # Higher weight for best performer
        "mednext_l": 0.35,   # Good generalization
        "custom_model": 0.25 # Specialized features
    }
    
    ensemble_pred = sum(
        predictions[arch] * arch_weights[arch] 
        for arch in architectures
    )
    
    return ensemble_pred, predictions
```

#### Multi-Fusion Ensemble
```python
fusion_types = [
    "masked_mean",       # Robust baseline
    "attention",         # Learned attention
    "channel_attention", # Channel-wise attention
    "uncertainty_weighted" # Uncertainty-based
]

def ensemble_multi_fusion(fusion_types, input_data):
    """
    Ensemble different fusion mechanisms
    """
    predictions = {}
    
    for fusion_type in fusion_types:
        model = load_fusion_model(fusion_type=fusion_type)
        pred = model(input_data)
        predictions[fusion_type] = pred
    
    # Fusion-specific confidence weighting
    confidences = {
        fusion_type: calculate_confidence(predictions[fusion_type])
        for fusion_type in fusion_types
    }
    
    # Weight by confidence
    total_confidence = sum(confidences.values())
    weighted_pred = sum(
        predictions[fusion_type] * confidences[fusion_type] / total_confidence
        for fusion_type in fusion_types
    )
    
    return weighted_pred, predictions
```

### 3. Sliding Window Inference

#### Basic Sliding Window
```python
def sliding_window_inference(model, volume, patch_size, overlap=0.5):
    """
    Process large volumes with sliding window approach
    """
    stride = tuple(int(p * (1 - overlap)) for p in patch_size)
    
    # Generate patch coordinates
    patches = generate_patch_coordinates(volume.shape, patch_size, stride)
    predictions = []
    
    for coords in patches:
        # Extract patch
        patch = extract_patch(volume, coords, patch_size)
        
        # Run inference
        with torch.no_grad():
            pred = model(patch)
        
        predictions.append((pred, coords))
    
    # Reconstruct full volume prediction
    full_pred = reconstruct_from_patches(predictions, volume.shape)
    
    return full_pred
```

#### Gaussian Weighted Sliding Window
```python
def gaussian_weighted_sliding_window(model, volume, patch_size, overlap=0.75):
    """
    Sliding window with Gaussian weighting for smooth predictions
    """
    # Create Gaussian weight map
    gaussian_weights = create_gaussian_weights(patch_size)
    
    # Initialize prediction and weight accumulators
    prediction_map = torch.zeros(volume.shape)
    weight_map = torch.zeros(volume.shape)
    
    stride = tuple(int(p * (1 - overlap)) for p in patch_size)
    patches = generate_patch_coordinates(volume.shape, patch_size, stride)
    
    for coords in patches:
        patch = extract_patch(volume, coords, patch_size)
        
        with torch.no_grad():
            pred = model(patch)
        
        # Apply Gaussian weighting
        weighted_pred = pred * gaussian_weights
        
        # Accumulate predictions and weights
        add_patch_to_map(prediction_map, weighted_pred, coords)
        add_patch_to_map(weight_map, gaussian_weights, coords)
    
    # Normalize by accumulated weights
    final_prediction = prediction_map / (weight_map + 1e-8)
    
    return final_prediction
```

### 4. Uncertainty Estimation

#### Monte Carlo Dropout
```python
def monte_carlo_inference(model, input_data, n_samples=50, dropout_rate=0.1):
    """
    Estimate prediction uncertainty using Monte Carlo dropout
    """
    # Enable dropout during inference
    model.train()
    set_dropout_rate(model, dropout_rate)
    
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(input_data)
            predictions.append(pred)
    
    # Calculate statistics
    predictions = torch.stack(predictions)
    mean_pred = torch.mean(predictions, dim=0)
    std_pred = torch.std(predictions, dim=0)
    
    # Uncertainty metrics
    epistemic_uncertainty = torch.mean(std_pred)
    confidence = 1.0 / (1.0 + epistemic_uncertainty)
    
    model.eval()  # Restore evaluation mode
    
    return {
        'prediction': mean_pred,
        'uncertainty': std_pred,
        'confidence': confidence,
        'all_samples': predictions
    }
```

#### Ensemble Disagreement
```python
def ensemble_disagreement_uncertainty(ensemble_predictions):
    """
    Measure uncertainty as disagreement between ensemble models
    """
    predictions = torch.stack(ensemble_predictions)
    
    # Mean prediction
    mean_pred = torch.mean(predictions, dim=0)
    
    # Disagreement measures
    variance = torch.var(predictions, dim=0)
    entropy = calculate_entropy(mean_pred)
    
    # Mutual information (epistemic uncertainty)
    individual_entropies = [calculate_entropy(pred) for pred in predictions]
    mean_entropy = torch.mean(torch.stack(individual_entropies))
    mutual_info = entropy - mean_entropy
    
    return {
        'prediction': mean_pred,
        'aleatoric_uncertainty': entropy,
        'epistemic_uncertainty': mutual_info,
        'total_uncertainty': variance,
        'confidence': 1.0 / (1.0 + entropy)
    }
```

#### Temperature Calibration
```python
def temperature_calibration(logits, temperature=1.0):
    """
    Apply temperature scaling for better probability calibration
    """
    calibrated_logits = logits / temperature
    probabilities = torch.softmax(calibrated_logits, dim=-1)
    
    return probabilities

def find_optimal_temperature(validation_logits, validation_labels):
    """
    Find optimal temperature parameter using validation data
    """
    from scipy.optimize import minimize_scalar
    
    def calibration_loss(temp):
        probs = temperature_calibration(validation_logits, temp)
        return -torch.mean(validation_labels * torch.log(probs + 1e-8))
    
    result = minimize_scalar(calibration_loss, bounds=(0.1, 10.0))
    return result.x
```

## Aggregation Strategies Configuration

### Configuration Examples

#### Basic TTA
```yaml
tta_config:
  enable: true
  transforms:
    - "original"
    - "flip_x"
    - "flip_y" 
    - "flip_z"
  aggregation: "mean"
  confidence_threshold: 0.1
```

#### Advanced Ensemble
```yaml
ensemble_config:
  enable: true
  methods:
    - type: "multi_model"
      checkpoints: 
        - "model1/best.ckpt"
        - "model2/best.ckpt"
      weights: [0.6, 0.4]
    - type: "multi_architecture"
      architectures: ["unet_xl", "mednext_l"]
      weights: "performance_based"
  uncertainty:
    enable: true
    method: "disagreement"
    threshold: 0.2
```

#### Production Pipeline
```yaml
production_config:
  tta:
    enable: true
    transforms: ["original", "flip_x", "flip_y"]
  ensemble:
    enable: false  # Single best model for speed
  sliding_window:
    enable: true
    overlap: 0.5
    gaussian_weights: true
  uncertainty:
    enable: true
    method: "monte_carlo"
    samples: 10
  performance:
    batch_size: 1
    optimize_memory: true
```

## Performance vs. Accuracy Trade-offs

| Method | Accuracy Gain | Time Overhead | Memory Overhead |
|--------|---------------|---------------|-----------------|
| Basic TTA (4 views) | +2-5% | 4x | 1x |
| Advanced TTA (8 views) | +3-7% | 8x | 1x |
| Multi-model Ensemble (3 models) | +5-10% | 3x | 3x |
| Multi-architecture Ensemble | +7-12% | 2-4x | 2-4x |
| Sliding Window (50% overlap) | +1-3% | 1.5x | 1x |
| Monte Carlo Dropout | +2-4% | 10-50x | 1x |
| Full Pipeline | +10-15% | 10-20x | 2-4x |

## Implementation Guidelines

### 1. Start Simple
```python
# Begin with basic inference
prediction = model(input_data)

# Add TTA for immediate improvement
if args.tta_enable:
    prediction = apply_tta_inference(model, input_data, tta_config)
```

### 2. Add Ensemble Gradually
```python
# Single model first
if len(ensemble_checkpoints) == 1:
    prediction = single_model_inference(input_data)
# Then ensemble
else:
    prediction = ensemble_inference(ensemble_checkpoints, input_data)
```

### 3. Monitor Performance
```python
# Track timing and memory usage
start_time = time.time()
prediction = inference_pipeline(input_data)
inference_time = time.time() - start_time

memory_used = torch.cuda.max_memory_allocated()
```

### 4. Optimize for Production
```python
# Profile-driven optimization
if production_mode:
    # Use fastest methods with acceptable accuracy loss
    config = load_production_config()
else:
    # Research mode: maximize accuracy
    config = load_research_config()
```

## Best Practices

### Accuracy Optimization
1. **Use diverse ensemble**: Different architectures and training strategies
2. **Apply TTA selectively**: Focus on most effective transformations
3. **Calibrate probabilities**: Use temperature scaling for better confidence
4. **Monitor uncertainty**: Filter uncertain predictions

### Performance Optimization  
1. **Profile bottlenecks**: Identify slow components
2. **Batch operations**: Process multiple cases together when possible
3. **Memory management**: Use gradient checkpointing and efficient data loading
4. **Model optimization**: Consider quantization and pruning for production

### Reliability Enhancement
1. **Validate on test set**: Measure actual performance gains
2. **Cross-validation**: Ensure robustness across data splits
3. **Error analysis**: Understand failure modes
4. **Monitoring**: Track performance in production

## Integration with Main Pipeline

The aggregation methods integrate seamlessly with the main inference pipeline:

```python
# In predict_task1.py
def main():
    args = parse_arguments()
    
    # Load base model
    model = load_model(args.checkpoint)
    
    # Configure aggregation
    if args.tta_enable or args.ensemble_checkpoints:
        aggregator = InferenceAggregator(
            base_model=model,
            tta_config=create_tta_config(args),
            ensemble_config=create_ensemble_config(args),
            uncertainty_config=create_uncertainty_config(args)
        )
        prediction = aggregator.predict(input_data)
    else:
        prediction = model(input_data)
    
    # Apply confidence thresholding
    if args.confidence_threshold > 0:
        prediction = apply_confidence_filter(prediction, args.confidence_threshold)
    
    # Save result
    save_output_txt(prediction, args.output)
```
