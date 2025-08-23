# Task 3 Comprehensive Implementation Plan

## Overview

This document outlines the step-by-step plan to address three critical requirements for Task 3 (Brain Age Regression):

1. **Fix Task3 Loss Bug** - Resolve tensor size mismatch warnings
2. **Inference Integration** - Add inference support for `fomo3_brain_age_256`
3. **Cross-Validation Support** - Implement 5-fold cross-validation

## 1. Fix Task3 Loss Bug

### Problem Analysis

The tensor size mismatch warning occurs during Task3 training:
```
UserWarning: Using a target size (torch.Size([2])) that is different to the input size (torch.Size([2, 1])).
```

This indicates that:
- **Target**: `torch.Size([2])` - Scalar age values per batch item
- **Input**: `torch.Size([2, 1])` - Model output with explicit feature dimension

### Root Cause Investigation

Based on code analysis in `src/models/supervised_reg.py`, the issue occurs in:

1. **Model Output**: Multi-encoder regression models output `[B, 1]` for regression tasks
2. **Target Shape**: Labels are loaded as `[B]` scalars (age values)
3. **Loss Function**: Expects matching dimensions for proper gradient computation

### Fix Strategy

#### Files to Modify:
- `src/models/supervised_reg.py` (primary fix)
- `src/models/networks/regression_head.py` (if exists)

#### Implementation Steps:

1. **Shape Normalization in Loss Computation**:
   - Ensure consistent tensor shapes before loss calculation
   - Add explicit shape handling in `validation_step()` and `training_step()`

2. **Output Shape Standardization**:
   - Standardize regression output to always be `[B]` for scalar targets
   - Update model architecture to output correct shapes

3. **Target Processing Enhancement**:
   - Ensure target tensors maintain correct dimensions throughout pipeline
   - Add validation for shape compatibility

#### Code Changes:

```python
# In src/models/supervised_reg.py
def validation_step(self, batch, _batch_idx):
    inputs, target, file_path = self._process_batch(batch)
    
    # Apply TTA or standard forward pass
    output = self._compute_tta_prediction(inputs) if self._val_tta_enable else self(inputs)
    
    # FIX: Ensure tensor shapes match for loss computation
    if output.dim() > 1 and output.size(-1) == 1:
        output = output.squeeze(-1)  # [B, 1] -> [B]
    
    # Ensure target is also 1D if needed
    if target.dim() > 1 and target.size(-1) == 1:
        target = target.squeeze(-1)  # [B, 1] -> [B]
    
    loss = self.loss_fn_val(output, target)
    # ... rest of method
```

### Testing Strategy

#### Unit Test: `test_task3_loss_bug_fix.py`

```python
import torch
import pytest
from models.supervised_reg import SupervisedRegModel
from data.task_configs import task3_config

class TestTask3LossBugFix:
    def test_tensor_shape_consistency(self):
        """Test that model output and target shapes are consistent for loss computation."""
        # Create mock config for Task3
        config = {
            **task3_config,
            "loss_type": "mae",
            "age_normalization": True,
            "age_mean": 61.87,
            "age_std": 15.09
        }
        
        model = SupervisedRegModel(config=config)
        
        # Mock batch with problematic shapes
        batch_size = 2
        mock_batch = {
            "image": torch.randn(batch_size, 2, 32, 32, 32),  # [B, C, D, H, W]
            "label": torch.tensor([65.0, 72.0]),  # [B] - scalar ages
            "file_path": ["subject_001", "subject_002"]
        }
        
        # Test that validation_step handles shapes correctly
        with torch.no_grad():
            # This should not raise tensor size mismatch warnings
            model.validation_step(mock_batch, 0)
    
    def test_model_output_shape_standardization(self):
        """Test that model outputs correct shape for regression."""
        config = {**task3_config, "loss_type": "mae"}
        model = SupervisedRegModel(config=config)
        
        # Test forward pass shape
        batch_size = 2
        inputs = torch.randn(batch_size, 2, 32, 32, 32)
        
        with torch.no_grad():
            output = model(inputs)
            
        # Output should be [B] for scalar regression
        assert output.shape == torch.Size([batch_size]), f"Expected {torch.Size([batch_size])}, got {output.shape}"
    
    def test_loss_computation_compatibility(self):
        """Test that loss functions work with corrected tensor shapes."""
        config = {**task3_config, "loss_type": "mae"}
        model = SupervisedRegModel(config=config)
        
        # Test all loss types
        for loss_type in ["mse", "mae", "huber"]:
            config["loss_type"] = loss_type
            model = SupervisedRegModel(config=config)
            
            output = torch.tensor([65.0, 72.0])  # [B]
            target = torch.tensor([63.0, 70.0])  # [B]
            
            # Should compute loss without warnings
            loss_fn_train, loss_fn_val = model._configure_losses()
            loss = loss_fn_val(output, target)
            
            assert torch.isfinite(loss), f"Loss should be finite for {loss_type}"
```

### Validation Approach

1. **Reproduce Bug**: Create test that triggers the warning before fix
2. **Validate Fix**: Ensure warning disappears after implementing shape fixes
3. **Regression Test**: Verify fix doesn't break existing functionality
4. **Integration Test**: Test with actual Task3 training pipeline

---

## 2. Inference Integration (Task3, fomo3_brain_age_256)

### Current State Analysis

The checkpoint is located at:
```
runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/checkpoints/best.ckpt
```

Configuration file:
```
runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/hparams.yaml
```

### Implementation Strategy

#### Files to Create/Modify:
- `src/inference/predict_task3.py` (new)
- `predict_task3.sh` (new shell script)
- `src/inference/predict.py` (enhance for Task3)

#### Core Features:

1. **Task3-Specific Prediction Script**
2. **Automatic Config Loading**
3. **Age Denormalization**
4. **Multi-modal Input Handling**
5. **CLI Interface Design**

### CLI Interface Design

#### Required Arguments:
- `--modalities`: Space-separated paths to T1 and T2 NIfTI files
- `--output_dir`: Directory to save predictions

#### Optional Arguments:
- `--checkpoint_path`: Override default checkpoint
- `--config_path`: Override default config
- `--device`: CUDA device ID (default: auto-detect)
- `--batch_size`: Inference batch size (default: 1)
- `--tta`: Enable test-time augmentation (default: False)
- `--age_denormalize`: Output denormalized age in years (default: True)

#### Usage Examples:

```bash
# Basic usage with default checkpoint
python src/inference/predict_task3.py \
  --modalities /path/to/t1.nii.gz /path/to/t2.nii.gz \
  --output_dir /path/to/outputs

# Advanced usage with custom settings
python src/inference/predict_task3.py \
  --modalities /path/to/t1.nii.gz /path/to/t2.nii.gz \
  --output_dir /path/to/outputs \
  --checkpoint_path /custom/path/best.ckpt \
  --device cuda:1 \
  --tta \
  --batch_size 2
```

### Shell Script Design: `predict_task3.sh`

```bash
#!/bin/bash
# FOMO Task 3 - Brain Age Prediction Inference Script

set -e

# Default configuration
DEFAULT_CHECKPOINT="runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/checkpoints/best.ckpt"
DEFAULT_CONFIG="runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/hparams.yaml"
DEFAULT_DEVICE="auto"
DEFAULT_BATCH_SIZE="1"
TTA_FLAG=""
OUTPUT_DIR=""
MODALITIES=()

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
    --device)
      DEFAULT_DEVICE="$2"
      shift 2
      ;;
    --batch_size)
      DEFAULT_BATCH_SIZE="$2"
      shift 2
      ;;
    --tta)
      TTA_FLAG="--tta"
      shift
      ;;
    -h|--help)
      echo "FOMO Task 3 - Brain Age Prediction"
      echo "Usage: $0 --modalities <t1.nii.gz> <t2.nii.gz> --output_dir <dir> [options]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Validation
if [ ${#MODALITIES[@]} -ne 2 ]; then
  echo "Error: Exactly 2 modalities required (T1 and T2)"
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: --output_dir is required"
  exit 1
fi

# Execute prediction
PYTHONPATH=src python src/inference/predict_task3.py \
  --modalities "${MODALITIES[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint_path "$DEFAULT_CHECKPOINT" \
  --config_path "$DEFAULT_CONFIG" \
  --device "$DEFAULT_DEVICE" \
  --batch_size "$DEFAULT_BATCH_SIZE" \
  $TTA_FLAG
```

### Implementation: `predict_task3.py`

Key components:

1. **Configuration Loading**:
   ```python
   def load_task3_config(config_path: str) -> Dict[str, Any]:
       """Load and process Task3 hyperparameters."""
       with open(config_path, 'r') as f:
           hparams = yaml.safe_load(f)
       return hparams['config']
   ```

2. **Age Denormalization**:
   ```python
   def denormalize_age(normalized_age: float, age_mean: float, age_std: float) -> float:
       """Convert normalized age back to years."""
       return normalized_age * age_std + age_mean
   ```

3. **Multi-modal Preprocessing**:
   ```python
   def preprocess_modalities(t1_path: str, t2_path: str, config: Dict) -> torch.Tensor:
       """Preprocess T1 and T2 modalities for inference."""
       # Load, resample, normalize according to training pipeline
   ```

### Unit Test: `test_task3_inference_integration.py`

```python
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock
from inference.predict_task3 import predict_brain_age, load_task3_config

class TestTask3InferenceIntegration:
    def test_config_loading(self):
        """Test loading of Task3 configuration from hparams.yaml."""
        # Mock hparams.yaml content
        mock_config = {
            'config': {
                'task_type': 'regression',
                'loss_type': 'mae',
                'age_normalization': True,
                'age_mean': 61.87,
                'age_std': 15.09
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(mock_config, f)
            config_path = f.name
        
        try:
            loaded_config = load_task3_config(config_path)
            assert loaded_config['task_type'] == 'regression'
            assert loaded_config['age_normalization'] == True
        finally:
            os.unlink(config_path)
    
    def test_age_denormalization(self):
        """Test age denormalization functionality."""
        from inference.predict_task3 import denormalize_age
        
        # Test with Task3 parameters
        normalized_age = 0.5  # Normalized age
        age_mean = 61.87
        age_std = 15.09
        
        denorm_age = denormalize_age(normalized_age, age_mean, age_std)
        expected = 0.5 * 15.09 + 61.87  # Should be ~69.4 years
        
        assert abs(denorm_age - expected) < 0.01
    
    @patch('inference.predict.ModelLoader.load_model')
    @patch('inference.predict.load_modalities')
    def test_brain_age_prediction_pipeline(self, mock_load_modalities, mock_load_model):
        """Test complete brain age prediction pipeline."""
        # Mock model that returns age prediction
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.5]])  # Normalized age
        mock_load_model.return_value = mock_model
        
        # Mock modality loading
        mock_load_modalities.return_value = [MagicMock(), MagicMock()]
        
        # Test prediction
        result = predict_brain_age(
            t1_path="mock_t1.nii.gz",
            t2_path="mock_t2.nii.gz",
            checkpoint_path="mock.ckpt",
            config={
                'age_normalization': True,
                'age_mean': 61.87,
                'age_std': 15.09
            }
        )
        
        # Should return denormalized age
        expected_age = 0.5 * 15.09 + 61.87
        assert abs(result - expected_age) < 0.01
```

---

## 3. Cross-Validation Support (5-fold)

### Current Implementation Analysis

The current K-fold implementation in `finetune.py` has basic support:
- `--k_folds`: Total number of folds
- `--fold_index`: Current fold index (0-based)

### Enhancement Strategy

#### Files to Modify:
- `run_fomo3_finetune_256_kfold.sh` (complete overwrite)
- `src/finetune.py` (enhance K-fold logic)
- `src/utils/cross_validation.py` (new utility module)

### K-fold Implementation Details

#### 1. Stratified Splitting Strategy

For regression tasks, we use **stratified binning** to ensure balanced age distribution across folds:

```python
def create_age_stratified_folds(subjects: List[str], ages: List[float], k: int = 5) -> List[Tuple[List[str], List[str]]]:
    """Create K stratified folds based on age quartiles."""
    # Bin ages into quartiles for stratification
    age_quartiles = np.percentile(ages, [25, 50, 75])
    
    def get_age_bin(age):
        if age <= age_quartiles[0]: return 0      # Q1
        elif age <= age_quartiles[1]: return 1    # Q2
        elif age <= age_quartiles[2]: return 2    # Q3
        else: return 3                            # Q4
    
    # Group subjects by age bin
    binned_subjects = {i: [] for i in range(4)}
    for subj, age in zip(subjects, ages):
        bin_idx = get_age_bin(age)
        binned_subjects[bin_idx].append(subj)
    
    # Create balanced folds
    folds = []
    for fold_idx in range(k):
        val_subjects = []
        train_subjects = []
        
        for bin_idx in range(4):
            bin_subjects = binned_subjects[bin_idx]
            bin_folds = [bin_subjects[i::k] for i in range(k)]
            val_subjects.extend(bin_folds[fold_idx])
            for i in range(k):
                if i != fold_idx:
                    train_subjects.extend(bin_folds[i])
        
        folds.append((train_subjects, val_subjects))
    
    return folds
```

#### 2. Sequential Execution Script

Enhanced `run_fomo3_finetune_256_kfold.sh`:

```bash
#!/bin/bash
# FOMO Task 3 - 5-Fold Cross-Validation Brain Age Regression

set -e

export CUDA_VISIBLE_DEVICES=1

echo "🧠 FOMO Task 3 - 5-Fold Cross-Validation Brain Age Regression"
echo "=============================================================="
echo "🎯 Task: Systematic 5-fold evaluation of brain age prediction"
echo "📊 Modalities: T1w + T2w with attention fusion"
echo "📈 Metrics: Pearson Correlation, MAE (years), R²"
echo ""

# Common training parameters
COMMON_ARGS="--taskid 3 \
  --data_dir /data/weidong/fomo-finetune \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --fusion_type attention \
  --modality_mapping T1=t1,T2=t2 \
  --t1_ckpt ckpt/t1.ckpt \
  --t2_ckpt ckpt/t2.ckpt \
  --precision bf16-mixed \
  --patch_size=256,256,32 \
  --train_batches_per_epoch=100 \
  --epochs 500 \
  --batch_size 2 \
  --num_devices 1 \
  --num_workers 0 \
  --starting_filters 64 \
  --freeze_encoder_epochs 4 \
  --phase1_head_lr 1e-3 \
  --phase2_head_lr 3e-4 \
  --phase2_encoder_lr 3e-5 \
  --label_smoothing 0.0 \
  --cls_head_dropout_p 0.2 \
  --loss_type mae \
  --age_normalization \
  --age_mean 61.87 \
  --age_std 15.09 \
  --grad_clip_val 1.0 \
  --grad_clip_algo norm \
  --num_sanity_val_steps 0 \
  --disable_early_stop \
  --augmentation_preset basic \
  --smoothing_window 10"

# Results tracking
RESULTS_FILE="./runs/Task003_FOMO3/kfold_results_summary.txt"
mkdir -p "$(dirname "$RESULTS_FILE")"
echo "FOMO Task 3 - 5-Fold Cross-Validation Results" > "$RESULTS_FILE"
echo "Generated: $(date)" >> "$RESULTS_FILE"
echo "=============================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Run all 5 folds
for fold in {0..4}; do
  echo "🔄 Running Fold $((fold + 1))/5..."
  echo "================================"
  
  PYTHONPATH=src /home/weidongguo/miniconda3/envs/fomo/bin/python src/finetune.py \
    $COMMON_ARGS \
    --k_folds 5 \
    --fold_index $fold \
    --experiment "fomo3_brain_age_256_kfold_fold${fold}"
  
  echo "✅ Completed Fold $((fold + 1))/5"
  echo ""
done

echo "📊 All folds completed! Computing aggregated metrics..."

# Aggregate results using Python script
PYTHONPATH=src python src/utils/aggregate_kfold_results.py \
  --results_dir ./runs/Task003_FOMO3/unet_xl \
  --experiment_pattern "fomo3_brain_age_256_kfold_fold*" \
  --output_file "$RESULTS_FILE" \
  --metrics "val/corr,val/mae"

echo ""
echo "🎉 5-Fold Cross-Validation Complete!"
echo "====================================="
echo "📈 Results summary saved to: $RESULTS_FILE"
echo ""
echo "🧠 Key Metrics to Review:"
echo "   • Pearson Correlation (val/corr): Mean ± Std"
echo "   • Mean Absolute Error (val/mae): Mean ± Std in years"
echo "   • Fold-wise consistency and variance"
echo ""
```

#### 3. Results Aggregation Utility

New file: `src/utils/aggregate_kfold_results.py`

```python
#!/usr/bin/env python
"""
K-Fold Cross-Validation Results Aggregation Utility

Aggregates metrics across K folds and computes statistics.
"""

import os
import glob
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

def extract_best_metrics_from_fold(fold_dir: str, metrics: List[str]) -> Dict[str, float]:
    """Extract best metrics from a single fold directory."""
    best_metrics_file = os.path.join(fold_dir, "best_metrics.txt")
    
    if not os.path.exists(best_metrics_file):
        return {}
    
    metrics_dict = {}
    with open(best_metrics_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                try:
                    metrics_dict[key.strip()] = float(value.strip())
                except ValueError:
                    continue
    
    # Extract requested metrics
    fold_metrics = {}
    for metric in metrics:
        if metric in metrics_dict:
            fold_metrics[metric] = metrics_dict[metric]
    
    return fold_metrics

def aggregate_kfold_results(results_dir: str, experiment_pattern: str, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Aggregate results across all K folds."""
    
    # Find all fold directories
    fold_dirs = glob.glob(os.path.join(results_dir, experiment_pattern, "version_0"))
    fold_dirs.sort()
    
    if len(fold_dirs) == 0:
        raise ValueError(f"No fold directories found matching pattern: {experiment_pattern}")
    
    print(f"Found {len(fold_dirs)} fold directories")
    
    # Collect metrics from each fold
    all_fold_metrics = []
    for fold_dir in fold_dirs:
        fold_metrics = extract_best_metrics_from_fold(fold_dir, metrics)
        if fold_metrics:
            all_fold_metrics.append(fold_metrics)
            print(f"Fold {len(all_fold_metrics)}: {fold_metrics}")
    
    if len(all_fold_metrics) == 0:
        raise ValueError("No valid metrics found across folds")
    
    # Compute aggregate statistics
    aggregated = {}
    for metric in metrics:
        values = [fold[metric] for fold in all_fold_metrics if metric in fold]
        
        if len(values) > 0:
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    return aggregated

def format_results_report(aggregated: Dict[str, Dict[str, float]]) -> str:
    """Format aggregated results into a readable report."""
    
    report = []
    report.append("AGGREGATED K-FOLD RESULTS")
    report.append("=" * 50)
    report.append("")
    
    for metric, stats in aggregated.items():
        report.append(f"📈 {metric.upper().replace('/', '_').replace('_', ' ').title()}:")
        report.append(f"   Mean: {stats['mean']:.4f}")
        report.append(f"   Std:  {stats['std']:.4f}")
        report.append(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Individual fold values
        report.append(f"   Fold Values: {[f'{v:.4f}' for v in stats['values']]}")
        report.append("")
    
    # Overall summary
    if 'val/corr' in aggregated and 'val/mae' in aggregated:
        corr_mean = aggregated['val/corr']['mean']
        mae_mean = aggregated['val/mae']['mean']
        report.append("🎯 CLINICAL INTERPRETATION:")
        report.append(f"   Average Correlation: {corr_mean:.3f} ({'Strong' if corr_mean > 0.8 else 'Moderate' if corr_mean > 0.6 else 'Weak'})")
        report.append(f"   Average Error: {mae_mean:.1f} years")
        report.append("")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Aggregate K-fold cross-validation results")
    parser.add_argument("--results_dir", required=True, help="Directory containing fold results")
    parser.add_argument("--experiment_pattern", required=True, help="Pattern to match experiment directories")
    parser.add_argument("--output_file", required=True, help="Output file for aggregated results")
    parser.add_argument("--metrics", required=True, help="Comma-separated list of metrics to aggregate")
    
    args = parser.parse_args()
    
    metrics = [m.strip() for m in args.metrics.split(',')]
    
    try:
        aggregated = aggregate_kfold_results(args.results_dir, args.experiment_pattern, metrics)
        report = format_results_report(aggregated)
        
        # Write to output file
        with open(args.output_file, 'a') as f:  # Append mode
            f.write("\n" + report + "\n")
        
        # Also print to console
        print(report)
        
    except Exception as e:
        print(f"Error aggregating results: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
```

### Unit Test: `test_cross_validation_support.py`

```python
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from utils.cross_validation import create_age_stratified_folds
from utils.aggregate_kfold_results import aggregate_kfold_results

class TestCrossValidationSupport:
    def test_age_stratified_fold_creation(self):
        """Test creation of age-stratified folds for regression."""
        # Mock subjects and ages
        subjects = [f"subject_{i:03d}" for i in range(100)]
        ages = list(range(20, 120))  # Ages 20-119
        
        folds = create_age_stratified_folds(subjects, ages, k=5)
        
        # Should have 5 folds
        assert len(folds) == 5
        
        # Each fold should have train/val split
        for train_subjects, val_subjects in folds:
            assert len(train_subjects) + len(val_subjects) == 100
            assert len(val_subjects) >= 15  # ~20% for validation
            assert len(val_subjects) <= 25  # ~20% for validation
            
            # No overlap between train and val
            assert len(set(train_subjects) & set(val_subjects)) == 0
    
    def test_age_distribution_balance(self):
        """Test that age distribution is balanced across folds."""
        subjects = [f"subject_{i:03d}" for i in range(100)]
        ages = [20, 30, 40, 50, 60, 70, 80, 90] * 12 + [25, 35, 45, 55]  # 100 subjects
        
        folds = create_age_stratified_folds(subjects, ages, k=5)
        
        # Check age distribution in each fold
        for fold_idx, (train_subjects, val_subjects) in enumerate(folds):
            # Get ages for validation subjects
            val_indices = [subjects.index(s) for s in val_subjects]
            val_ages = [ages[i] for i in val_indices]
            
            # Should have representation from different age ranges
            age_quartiles = [
                sum(1 for age in val_ages if age <= 40),    # Young
                sum(1 for age in val_ages if 40 < age <= 60),  # Middle
                sum(1 for age in val_ages if age > 60)      # Older
            ]
            
            # No quartile should be empty (balanced distribution)
            assert all(q > 0 for q in age_quartiles), f"Fold {fold_idx} has unbalanced age distribution"
    
    def test_kfold_results_aggregation(self):
        """Test aggregation of K-fold results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock fold result directories
            for fold in range(5):
                fold_dir = os.path.join(temp_dir, f"experiment_fold{fold}", "version_0")
                os.makedirs(fold_dir)
                
                # Create mock best_metrics.txt
                metrics_file = os.path.join(fold_dir, "best_metrics.txt")
                with open(metrics_file, 'w') as f:
                    f.write(f"val/corr: {0.8 + fold * 0.02}\n")  # 0.80, 0.82, 0.84, 0.86, 0.88
                    f.write(f"val/mae: {5.0 - fold * 0.2}\n")    # 5.0, 4.8, 4.6, 4.4, 4.2
            
            # Test aggregation
            aggregated = aggregate_kfold_results(
                results_dir=temp_dir,
                experiment_pattern="experiment_fold*",
                metrics=["val/corr", "val/mae"]
            )
            
            # Verify aggregated statistics
            assert "val/corr" in aggregated
            assert "val/mae" in aggregated
            
            # Check correlation statistics
            corr_stats = aggregated["val/corr"]
            assert abs(corr_stats["mean"] - 0.84) < 0.01  # Mean should be 0.84
            assert corr_stats["min"] == 0.80
            assert corr_stats["max"] == 0.88
            
            # Check MAE statistics  
            mae_stats = aggregated["val/mae"]
            assert abs(mae_stats["mean"] - 4.6) < 0.01    # Mean should be 4.6
            assert mae_stats["min"] == 4.2
            assert mae_stats["max"] == 5.0
```

---

## Implementation Timeline

### Phase 1: Loss Bug Fix (Days 1-2)
1. Analyze and reproduce the tensor size mismatch bug
2. Implement shape normalization fixes in `supervised_reg.py`
3. Create and run unit tests
4. Validate fix with integration tests

### Phase 2: Inference Integration (Days 3-4)
1. Create `predict_task3.py` with CLI interface
2. Implement shell script wrapper `predict_task3.sh`
3. Add configuration loading and age denormalization
4. Create unit tests for inference pipeline
5. Test with actual checkpoints

### Phase 3: Cross-Validation Support (Days 5-6)
1. Enhance K-fold logic in `finetune.py`
2. Create age-stratified folding utility
3. Implement results aggregation script
4. Overwrite `run_fomo3_finetune_256_kfold.sh`
5. Create comprehensive unit tests
6. Run validation with subset of data

### Phase 4: Integration Testing (Day 7)
1. End-to-end testing of all components
2. Performance validation
3. Documentation updates
4. Final regression testing

## Success Criteria

### Task 1: Loss Bug Fix
- [ ] No tensor size mismatch warnings during Task3 training
- [ ] All existing functionality preserved
- [ ] Unit tests pass with 100% coverage
- [ ] Integration tests confirm bug resolution

### Task 2: Inference Integration  
- [ ] CLI interface works with both shell script and Python script
- [ ] Age predictions are properly denormalized to years
- [ ] Automatic checkpoint/config detection works
- [ ] TTA support functional for improved accuracy
- [ ] Comprehensive error handling and validation

### Task 3: Cross-Validation Support
- [ ] 5-fold cross-validation runs successfully
- [ ] Age-stratified folding ensures balanced age distribution
- [ ] Results aggregation provides meaningful statistics
- [ ] Sequential execution completes without errors
- [ ] Fold-wise results are properly tracked and stored

## Risk Mitigation

### Technical Risks:
1. **Memory Issues**: Monitor GPU memory usage during K-fold training
2. **Checkpoint Compatibility**: Ensure inference works with existing checkpoints
3. **Shape Inconsistencies**: Thorough testing of tensor shapes across different batch sizes

### Mitigation Strategies:
1. **Incremental Testing**: Test each component individually before integration
2. **Backup Checkpoints**: Preserve original checkpoints before modifications  
3. **Comprehensive Validation**: Unit tests + integration tests + end-to-end tests
4. **Documentation**: Clear usage examples and troubleshooting guides

---

This comprehensive plan addresses all three requirements with detailed implementation strategies, robust testing approaches, and clear success criteria. Each component is designed to integrate seamlessly with the existing codebase while providing enhanced functionality for Task3 brain age regression.
