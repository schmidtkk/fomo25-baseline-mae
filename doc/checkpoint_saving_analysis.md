# FOMO Task Checkpoint Saving Analysis

## Overview
This document analyzes how the fine-tuning scripts for FOMO tasks determine when to save checkpoints, specifically focusing on the indicator metrics and saving logic.

## Checkpoint Saving Logic

### Monitor Metric Determination
The fine-tuning script (`src/finetune.py`) determines the monitor metric based on task type as follows:

```python
# Choose monitor metric based on task type
if task_type == "classification" and num_classes == 2:
    monitor_metric = "val/auroc_subject"
    monitor_mode = "max"
elif task_type == "regression":
    monitor_metric = "val/corr"  # Use correlation for brain age regression
    monitor_mode = "max"
else:
    monitor_metric = "val/loss"
    monitor_mode = "min"
```

### Task-Specific Monitor Metrics

#### Task 1 (FOMO1): Binary Classification
- **Monitor Metric**: `val/auroc_subject`
- **Mode**: `max` (higher is better)
- **Type**: Area Under the ROC Curve for subject-level predictions
- **Secondary Metrics**: `val/loss`, `val/accuracy`, `val/f1`, `val/precision`, `val/recall`

#### Task 2 (FOMO2): Segmentation
- **Monitor Metric**: `val/loss` 
- **Mode**: `min` (lower is better)
- **Type**: Validation loss (typically Dice loss for segmentation)
- **Secondary Metrics**: `val/dice`, `val/nsd` (Normal Surface Distance)

#### Task 3 (FOMO3): Brain Age Regression
- **Monitor Metric**: `val/corr` 
- **Mode**: `max` (higher is better)
- **Type**: Pearson Correlation Coefficient between predicted and actual ages
- **Secondary Metrics**: `val/loss`, `val/mae` (Mean Absolute Error)

## FOMO Task 3 Detailed Analysis

### Metrics Computation
The regression model (`src/models/supervised_reg.py`) computes metrics as follows:

1. **Correlation (`val/corr`)**:
   - Computed using PyTorch's `PearsonCorrCoef` metric
   - Applied to normalized age values for numerical stability
   - Logged at epoch level in `on_validation_epoch_end()`
   - Handles NaN/Inf values by replacing with 0.0

2. **Mean Absolute Error (`val/mae`)**:
   - Computed on denormalized age values for interpretability
   - Shows actual age prediction error in years
   - Uses age normalization: `age_mean=61.87, age_std=15.09`

3. **Loss (`val/loss`)**:
   - Uses MAE loss (`loss_type: mae`) for robustness to outliers
   - Applied to normalized age values during training
   - Alternative loss types: MSE, Huber

### Checkpoint Callback Configuration
The Enhanced ModelCheckpoint callback is configured as:

```python
checkpoint_callback = EnhancedModelCheckpoint(
    monitor=monitor_metric,  # "val/corr" for Task3
    mode=monitor_mode,       # "max" for Task3
    save_top_k=1,
    filename="best",
    enable_version_counter=False,
)
```

### Current Validation Example
From the training log analysis of `fomo3_brain_age_256_kfold_fold0`:

- **Monitor Metric**: `val/corr` (Pearson Correlation)
- **Best Performance**: ~0.69 correlation achieved around epoch 4-5
- **MAE Performance**: ~9-10 years mean absolute error
- **Age Normalization**: Applied with μ=61.87, σ=15.09

## Enhanced Metrics Tracking

### Current Best Metrics File
The `EnhancedModelCheckpoint` callback already creates a `best_metrics.txt` file that tracks:
- Best validation loss
- Best AUROC (for classification)
- Best correlation (for regression) 
- Best accuracy
- Best MAE/MSE
- Epoch and step information for each best metric

### File Location
Best metrics are saved to: `{checkpoint_dir}/best_metrics.txt`

For example: `runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256_kfold_fold0/version_0/best_metrics.txt`

### Checkpoint Saving Trigger
A new checkpoint is saved when:
1. The monitor metric improves (e.g., `val/corr` increases for Task3)
2. The `EnhancedModelCheckpoint._save_checkpoint()` method is called
3. Best metrics file is automatically updated with new best values

## Current Status Assessment

The current implementation already provides:
✅ Task-specific monitor metrics (val_mae assumption was incorrect - it's val/corr)
✅ Enhanced checkpoint callback with terminal feedback
✅ Best metrics tracking across all important metrics
✅ Persistent best_metrics.txt file with epoch/step information
✅ Comprehensive logging of training statistics

## Findings

### Corrected Assumptions
The initial assumption about Task3 using `val_mae` as the monitor metric was **incorrect**. The actual monitor metrics are:

- **Task1**: `val/auroc_subject` (not val_loss)
- **Task2**: `val/loss` (correct, but represents Dice loss for segmentation)
- **Task3**: `val/corr` (not val_mae - correlation is the primary metric)

### Current Implementation Quality
The existing checkpoint saving system is already quite sophisticated:

1. **Task-aware metric selection**: Automatically chooses appropriate metrics
2. **Enhanced callback**: Provides detailed terminal feedback and persistent logging
3. **Multi-metric tracking**: Monitors all relevant metrics, not just the primary one
4. **Robust handling**: Manages NaN/Inf values and edge cases
5. **Interpretable metrics**: Uses denormalized values for MAE (actual years)

The enhancement requirement is **already implemented** through the `EnhancedModelCheckpoint` callback and accompanying `best_metrics.txt` file.
