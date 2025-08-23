# Task 3 Implementation Plan - Quick Reference

## Overview
**Objective**: Address 3 critical requirements for Task3 (Brain Age Regression)

## 🐛 Task 1: Fix Loss Bug 
**Problem**: `UserWarning: Using a target size (torch.Size([2])) that is different to the input size (torch.Size([2, 1]))`

**Solution**: Shape normalization in `src/models/supervised_reg.py`
```python
# Fix in validation_step() and training_step()
if output.dim() > 1 and output.size(-1) == 1:
    output = output.squeeze(-1)  # [B, 1] -> [B]
```

**Test**: `src/tests/test_task3_loss_bug_fix.py`

---

## 🔮 Task 2: Inference Integration
**Target**: `runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/`

**Deliverables**:
- `src/inference/predict_task3.py` - Python inference script
- `predict_task3.sh` - Shell script wrapper

**Usage**:
```bash
# Basic usage
./predict_task3.sh --modalities t1.nii.gz t2.nii.gz --output_dir ./results

# Advanced usage  
./predict_task3.sh --modalities t1.nii.gz t2.nii.gz --output_dir ./results --tta --device cuda:1
```

**Test**: `src/tests/test_task3_inference_integration.py`

---

## 📊 Task 3: Cross-Validation (5-fold)
**Enhancement**: Age-stratified 5-fold CV with sequential execution

**Key Files**:
- `run_fomo3_finetune_256_kfold.sh` (overwrite) - Runs all 5 folds sequentially
- `src/utils/cross_validation.py` (new) - Stratified folding utilities
- `src/utils/aggregate_kfold_results.py` (new) - Results aggregation

**Execution**:
```bash
./run_fomo3_finetune_256_kfold.sh  # Runs folds 0-4 automatically
```

**Test**: `src/tests/test_cross_validation_support.py`

---

## 📋 Implementation Timeline (7 days)

| Days | Task | Deliverables |
|------|------|-------------|
| 1-2  | Loss Bug Fix | Shape normalization + tests |
| 3-4  | Inference Integration | CLI interface + shell script |
| 5-6  | Cross-Validation | 5-fold support + aggregation |
| 7    | Integration Testing | End-to-end validation |

---

## ✅ Success Criteria

**Task 1**: ✅ No tensor warnings + existing functionality preserved  
**Task 2**: ✅ CLI works + age denormalization + TTA support  
**Task 3**: ✅ 5-fold runs + stratified splits + results aggregation

---

## 🎯 Key Technical Features

### Loss Bug Fix
- Tensor shape normalization: `[B,1] → [B]`
- Preserve all existing functionality
- Support all loss types: MSE, MAE, Huber

### Inference Integration  
- Auto-detect checkpoints & configs
- Age denormalization: normalized → years
- TTA support for improved accuracy
- Robust error handling

### Cross-Validation
- Age-stratified folding (balanced distribution)
- Sequential execution of all 5 folds
- Automatic results aggregation with statistics
- Clinical interpretation thresholds

---

**Full Details**: See `doc/task3_comprehensive_implementation_plan.md`
