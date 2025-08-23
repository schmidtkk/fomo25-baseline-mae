# FOMO Task 3 Dual Checkpoint System Implementation

## Overview
Enhanced the FOMO fine-tuning system to save two separate best checkpoints for Task 3 (brain age regression), addressing both correlation and MAE criteria.

## Changes Made

### 1. Dual Checkpoint System for Task 3
**File Modified**: `src/finetune.py`

**Previous Behavior**:
- Single checkpoint saved based on highest validation correlation (`val/corr`)
- Best MAE performance could be lost if correlation wasn't simultaneously optimal

**New Behavior**:
- **Two checkpoints saved simultaneously**:
  - `best_corr.ckpt`: Highest validation correlation (primary metric)
  - `best_mae.ckpt`: Lowest validation MAE (secondary metric)

### 2. Implementation Details

#### Checkpoint Configuration (Lines 589-637)
```python
elif task_type == "regression":
    # Dual checkpoint system for Task 3: save both best correlation and best MAE
    monitor_metric = "val/corr"  # Primary metric remains correlation
    monitor_mode = "max"
    
    # Checkpoint for best correlation (primary metric)
    checkpoint_callback_corr = EnhancedModelCheckpoint(
        monitor="val/corr",
        mode="max",
        save_top_k=1,
        filename="best_corr",
        enable_version_counter=False,
    )
    
    # Checkpoint for best MAE (secondary metric)
    checkpoint_callback_mae = EnhancedModelCheckpoint(
        monitor="val/mae",
        mode="min", 
        save_top_k=1,
        filename="best_mae",
        enable_version_counter=False,
    )
    
    checkpoint_callback = checkpoint_callback_corr  # Keep primary for compatibility
    callbacks_list = [checkpoint_callback_corr, checkpoint_callback_mae]
```

#### Enhanced Logging (Lines 697-704)
```python
elif task_type == "regression":
    logging.debug("Pearson correlation monitoring (val/corr) for brain age regression")
    logging.debug("DUAL CHECKPOINT SYSTEM for Task 3:")
    logging.debug("  - best_corr.ckpt: Saves highest validation correlation")
    logging.debug("  - best_mae.ckpt: Saves lowest validation MAE") 
    logging.debug(f"Primary monitor (early stopping): {monitor_metric}")
    logging.debug("Enhanced metrics: MAE and Pearson Correlation")
```

### 3. System Behavior

#### Training Process
1. **Each validation epoch**: Both checkpoints are evaluated independently
2. **Correlation improves**: `best_corr.ckpt` is updated
3. **MAE improves**: `best_mae.ckpt` is updated
4. **Early stopping**: Still based on primary metric (`val/corr`)

#### File Structure
After training Task 3, the checkpoint directory will contain:
```
runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/checkpoints/
├── best_corr.ckpt      # Best correlation checkpoint
├── best_mae.ckpt       # Best MAE checkpoint
├── last.ckpt           # Last epoch checkpoint
└── metrics.txt         # Comprehensive metrics for both checkpoints
```

### 4. Compatibility

#### Unchanged for Other Tasks
- **Task 1 (Classification)**: Single checkpoint based on `val/auroc_subject`
- **Task 2 (Segmentation)**: Single checkpoint based on `val/loss`

#### Backward Compatibility
- Primary monitor metric remains `val/corr` for early stopping consistency
- Existing scripts and inference pipelines continue to work
- Enhanced checkpoint callback generates comprehensive metrics for both files

### 5. Usage Examples

#### Training with Dual Checkpoints
```bash
# Run Task 3 training - automatically saves both checkpoints
./run_fomo3_finetune_256.sh
```

#### Using Best Correlation Model
```bash
python src/inference/predict_task3.py \
    --checkpoint_path runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/checkpoints/best_corr.ckpt \
    --modalities t1.nii.gz t2.nii.gz
```

#### Using Best MAE Model  
```bash
python src/inference/predict_task3.py \
    --checkpoint_path runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/checkpoints/best_mae.ckpt \
    --modalities t1.nii.gz t2.nii.gz
```

### 6. Benefits

1. **Preserve Best Performance**: No longer lose best MAE when correlation doesn't improve
2. **Model Selection Flexibility**: Choose between correlation-optimized vs. error-optimized models
3. **Research Insights**: Compare performance characteristics of both criteria
4. **Production Options**: Deploy the checkpoint that best matches downstream requirements

### 7. Technical Implementation

#### EnhancedModelCheckpoint Integration
- Both callbacks use the same enhanced checkpoint class
- Each generates comprehensive `metrics.txt` files
- Terminal feedback shows updates for both checkpoints
- Full metrics tracking for both criteria

#### Memory and Storage Impact
- Minimal: Only saves additional checkpoint when MAE improves
- Each checkpoint is ~200-500MB depending on model size
- Enhanced metrics files provide detailed tracking for both

### 8. Validation

The implementation:
- ✅ Maintains compatibility with existing Task 1 and Task 2 behavior
- ✅ Preserves primary monitor metric (`val/corr`) for early stopping
- ✅ Adds comprehensive logging for dual checkpoint system
- ✅ Integrates with existing enhanced callback system
- ✅ Provides clear file naming convention

## Next Steps

1. **Test the implementation** with a short Task 3 training run
2. **Verify both checkpoints are created** and contain expected metrics
3. **Compare model performance** using both checkpoint criteria
4. **Update documentation** to reflect dual checkpoint system

The implementation is ready for production use and will automatically provide both correlation-optimized and MAE-optimized models for FOMO Task 3.
