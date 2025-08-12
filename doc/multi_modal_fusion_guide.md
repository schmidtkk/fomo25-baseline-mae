## Multi-Modal Encoder Fusion: Developer Guide

### Overview
- Finetuning with multiple modality-specific encoders fused at each scale.
- Robust to missing modalities, numerically stable fusion.
- Compatible with modal/group-specific self-supervised pretraining.

- **Global modality vocabulary**: `["t1", "t2", "flair", "dwi", "other"]`
- **FOMO1 canonical modalities**: `("DWI", "ADC", "T2FLAIR", "SWI_OR_T2STAR")`
- **Default model**: `unet_xl`

### Key Components
- `src/models/multiencoder.py`: wraps per-modality encoders and applies fusion.
- `src/models/fusion/masked_mean.py`: Masked Mean Fusion layer.
- `src/models/networks/unet.py`: UNet with optional multi-encoder encoder.
- `src/data/preprocess/fomo1_fusion.py`: per-modality preprocessing for FOMO1.
- `src/data/dataset_fusion.py`: dataset for per-modality input and mask.
- `src/finetune.py`: CLI for finetune, dataset selection, weight loading.

## Pretraining
- `src/pretrain.py` supports modal-specific pretraining via `--modality-mode` to produce checkpoints for groups in the global vocabulary (e.g., `dwi`, `flair`).
- You can skip sanity validation with:
  ```bash
  --skip_sanity_check
  ```
- Ensure correct GPU visibility; avoid hiding the only GPU (remove incorrect `CUDA_VISIBLE_DEVICES`).

## Data Preprocessing (FOMO1 Fusion)
- Canonical order: `DWI`, `ADC`, `T2FLAIR`, `SWI_OR_T2STAR` (prefer SWI over T2STAR).
- Input layout (example):
  - `.../fomo-task1/preprocessed/<subject>/ses_1/*.nii.gz`
  - `.../fomo-task1/labels/<subject>/ses_1/label.txt`
- Output layout (example):
  - `.../Task001_FOMO1_fusion/<subject>/{DWI.npy,ADC.npy,T2FLAIR.npy,SWI_OR_T2STAR.npy,mask.json,label.txt}`

- Run:
  ```bash
  PYTHONPATH=src python src/data/preprocess/fomo1_fusion.py \
    --source_path /abs/path/to/fomo-task1 \
    --output_path /abs/path/parent/output \
    --num_workers $(nproc)
  ```

- Implementation notes:
  - Handles different `yucca` return types (list or ndarray); ensures `[N_present, D, H, W]`.
  - Writes `mask.json` with presence flags per canonical modality.
  - Missing modalities are allowed and recorded.

## Data Preprocessing (FOMO3 Fusion)
- Canonical order: `T1`, `T2`.
- Input layout (example):
  - `.../fomo-task3/preprocessed/<subject>/ses_1/*.nii.gz`
  - `.../fomo-task3/labels/<subject>/ses_1/label.txt` (age)
- Output layout (example):
  - `.../Task003_FOMO3_fusion/<subject>/{T1.npy,T2.npy,mask.json,label.txt}`

- Run:
  ```bash
  PYTHONPATH=src python src/data/preprocess/fomo3_fusion.py \
    --source_path /abs/path/to/fomo-task3 \
    --output_path /abs/path/parent/output \
    --num_workers $(nproc)
  ```

## Dataset
- `FusionCLSDataset` (`src/data/dataset_fusion.py`):
  - Loads `[M, D, H, W]` with `M=4` for FOMO1 canonical.
  - Returns `modality_mask` `[M]` (1=present, 0=missing).
  - Zero-fills missing channels if `allow_missing_modalities=True`.

- Auto-selection in `src/finetune.py`:
  - If `--data_dir/<TaskName>_fusion` exists, `FusionCLSDataset` is used automatically.

## Model and Fusion
- Multi-encoder UNet (`src/models/networks/unet.py`):
  - `--use_multi_encoder` builds a `MultiModalEncoderWithFusion`.
  - Otherwise, a single `UNetEncoder` is used.

- Multi-encoder wrapper (`src/models/multiencoder.py`):
  - Creates an encoder per finetune modality.
  - Maps finetune modalities to global groups to compute `modality_group_ids`.

- Masked Mean Fusion (`src/models/fusion/masked_mean.py`):
  - Pre-alignment per modality: `GroupNorm → 1x1x1 Conv`.
  - Learnable per-global-group `gamma`: shape `[len(global_vocab), C]`.
  - Masked mean with `clamp_min(1)` for numerical stability (all-missing safe).
  - Learnable post-fusion scale and bias.

## Weight Loading Strategy
- Global vocabulary: `["t1", "t2", "flair", "dwi", "other"]`.
- Default finetune→pretrain mapping for FOMO1:
  - `DWI→dwi`, `ADC→dwi`, `T2FLAIR→flair`, `SWI_OR_T2STAR→other`.
- Custom mapping:
  ```bash
  --modality_mapping "DWI=dwi,ADC=dwi,T2FLAIR=flair,SWI_OR_T2STAR=other"
  ```

- Provide per-group checkpoints with:
  ```bash
  --modality_ckpts "dwi=/abs/dwi.ckpt,flair=/abs/flair.ckpt,other=/abs/other.ckpt"
  ```
- Missing checkpoint tolerance:
  - If a mapped group has no checkpoint provided, that modality’s encoder remains randomly initialized (warning printed).
  - No implicit `all` fallback is used.

- Single-encoder path:
  - Use `--pretrained_weights_path` to load a single checkpoint with `strict=False`.
  - Heads use `nn.LazyLinear` to adapt to varying bottleneck sizes.

## Finetune: How to Run
- Fusion data parent dir example: `/mnt/.../FOMO-MRI/fomo-finetune` containing `Task001_FOMO1_fusion`.
- Explicit interpreter prevents PATH/env issues:
  ```bash
  PY=/mnt/cvlab/scratch/cvlab/home/hantzhan/anaconda3/envs/fomo/bin/python
  cd /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main

  PYTHONPATH=src "$PY" src/finetune.py \
    --taskid 1 \
    --data_dir /mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune \
    --save_dir ./runs \
    --model_name unet_xl \
    --use_multi_encoder \
    --modality_ckpts "dwi=/abs/path/dwi.ckpt,flair=/abs/path/flair.ckpt,other=/abs/path/other.ckpt" \
    --epochs 100 --batch_size 2 --num_devices 1 --num_workers 8 --new_version
  ```
- Precision:
  - If BF16 unsupported, add `--precision 16-mixed`.

### K-Fold controls and metrics (Task 1)
- Set total folds and current fold via `--k_folds K` and `--fold_index f` (0-based). Splits are stratified at subject level in fusion mode.
- Validation logs a subject-level AUROC (`val/auroc_subject`) by aggregating per-subject predictions (mean logit) before AUROC.

### K-Fold ensembling (Task 1)
- Utilities in `src/utils/ensemble.py`:
  - `average_probabilities([...], weights=None)`: average per-sample probabilities across folds/models.
  - `average_subject_probabilities([...], weights=None)`: average per-subject positive-class probs across folds.
  - `auroc_from_subject_probabilities(subject_to_prob, subject_to_target)`: compute AUROC at subject level.
- CLI helper: `src/utils/ensemble_cli.py` supports ensembling multiple JSON files with schema:
  ```json
  {
    "subject_probs": {"FOMO1_A": 0.73, "FOMO1_B": 0.21},
    "subject_targets": {"FOMO1_A": 1, "FOMO1_B": 0}
  }
  ```
  Example:
  ```bash
  PYTHONPATH=src python src/utils/ensemble_cli.py \
    --inputs runs/fomo1_k3/fold0/.../subject_probs/val_subject_probs_epoch_0012.json \
            runs/fomo1_k3/fold1/.../subject_probs/val_subject_probs_epoch_0010.json \
            runs/fomo1_k3/fold2/.../subject_probs/val_subject_probs_epoch_0011.json \
    --out runs/fomo1_k3/ensemble_subject_probs.json
  ```
  It writes the averaged `subject_probs` and prints AUROC if `subject_targets` are present.
- Cross-validation summary tool: `tools/summarize_kfold.py` automatically locates the latest `subject_probs` JSON per fold and reports per-fold AUROC and ensembled AUROC.
  ```bash
  PYTHONPATH=src python tools/summarize_kfold.py runs/fomo1_k3 --num_folds 3 --print_paths
  ```

### Subject-level export and optimization options
- Subject-level JSON export (for ensembling/analysis): enable via `--export_subject_probs`. Files are written under `version_dir/subject_probs/` each validation epoch with schema:
```json
{
  "subject_probs": {"FOMO1_A": 0.73, "FOMO1_B": 0.21},
  "subject_targets": {"FOMO1_A": 1, "FOMO1_B": 0},
  "epoch": 12
}
```
- Training stability flags in `src/finetune.py`:
  - `--accumulate_grad_batches N`
  - `--grad_clip_val V`
  - `--channels_last`
  - `--use_swa --swa_lrs LR`
  - `--use_ema --ema_decay 0.999`
    - EMA swaps into validation automatically and the best EMA checkpoint is saved as `ema-best.ckpt` under `.../checkpoints/`.
  - Schedulers and LR policy (optional, defaults off or cosine):
    - `--scheduler {cosine, cosine_restarts, one_cycle, none}`; `--one_cycle_max_lr`.
    - Layer-wise LR decay: `--apply_layerwise_lr_decay`, `--layerwise_lr_decay_gamma`.
- Validation TTA:
  - `--val_tta` enables a light placeholder TTA path. For full TTA, integrate input flips in forward.
- Per-subject CSV export:
  - Alongside JSON, a CSV is written per epoch to `version_dir/subject_probs/val_subject_probs_epoch_XXXX.csv`.
  - Writer utility: `src/utils/logging_utils.py`.

### Calibration
- Module: `src/utils/calibration.py`
- `TemperatureScaler`: fits a single temperature on validation probs/targets using NLL; provides:
  - `fit_from_probs(probs, targets)` → learned temperature
  - `forward_probs(probs)` → calibrated probabilities
  - `brier_score(probs, targets)`
- Unit test: `src/tests/test_calibration.py`.

### Loss and imbalance options (classification)
- Module: `src/utils/losses.py`
- Flags via model `config` (set in `finetune.py` before model creation):
  - `class_weights`: list of per-class weights (e.g., `[w_neg, w_pos]` for binary) used for CE or to derive `pos_weight` for BCE.
  - `focal_gamma` and optional `focal_alpha`: enable focal loss (binary or multiclass).
  - `label_smoothing`: applied to training CE for multiclass; validation uses 0 smoothing.
- Tests: `src/tests/test_losses.py`.

## New features (Task 1 finetune)

- Fusion vs stacked switch
  - `--fusion_mode fusion`: force per-modality fusion dataset (e.g., `Task001_FOMO1_fusion`) and enable multi-encoder.
  - `--fusion_mode stacked`: legacy single `.npy` dataset and single-encoder.
  - `--fusion_mode auto`: detect fusion folder and choose automatically (default).

- Checkpoint flags (fusion and stacked)
  - Fusion (per-group ckpts):
    - `--dwi_ckpt`, `--flair_ckpt`, `--t1_ckpt`, `--t2_ckpt`, `--other_ckpt`
    - Optional legacy mapping: `--modality_ckpts "dwi=/abs/dwi.ckpt,flair=/abs/flair.ckpt,..."` (no `all` fallback)
    - Missing groups initialize randomly; this is expected and supported.
  - Stacked (single-encoder):
    - `--all_ckpt` (preferred) or `--pretrained_weights_path`

- K-Fold controls
  - `--k_folds K` and `--fold_index f` (0-based). In fusion mode, folds are stratified at subject-level (no leakage).
  - Recommended for Task 1: K=3 (fallback K=2 if balance breaks).

- Subject-level AUROC (Task 1)
  - Validation logs `val/auroc_subject` computed by averaging multiple predictions per subject (if any) to a single logit, then computing AUROC.
  - This metric is used for early stopping and best-checkpoint selection in binary classification.

- Early stopping and checkpoint monitoring
  - `--early_stop_patience` (default 12), `--early_stop_min_delta` (default 0.002)
  - Monitored metric auto-selects `val/auroc_subject` for binary classification (else `val/loss`); best checkpoint saved as `best.ckpt`.

- Two-phase finetune schedule
  - `--freeze_encoder_epochs` (default 15): freeze encoders initially and train the head.
  - LR controls:
    - `--phase1_head_lr` (default 5e-4) during freeze phase.
    - `--phase2_head_lr` (default 2e-4) and `--phase2_encoder_lr` (default 1e-5) after unfreezing.
  - Cosine scheduler with 5-epoch warmup to 0.1× by end of training (matches repo default style).

- Runtime safety and memory

- K-Fold ensemble utilities
  - `src/utils/ensemble.py` implements probability and subject-level ensembling.
  - `src/utils/ensemble_cli.py` provides a minimal CLI to ensemble per-fold subject probabilities.
  - Use `--precision 16-mixed`, `--batch_size 1–2`, and `--patch_size 24–32` (divisible by 8). If still limited, try `--model_name unet_b`.

### Example: Task 1 finetune with fusion, K-fold, EMA, subject export
```bash
PY=/mnt/cvlab/scratch/cvlab/home/hantzhan/anaconda3/envs/fomo/bin/python
cd /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main

PYTHONPATH=src "$PY" src/finetune.py \
  --taskid 1 \
  --data_dir /mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --k_folds 3 --fold_index 0 \
  --dwi_ckpt   /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/dwi.ckpt \
  --flair_ckpt /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/flair.ckpt \
  --t1_ckpt    /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/t1.ckpt \
  --t2_ckpt    /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/t2.ckpt \
  --early_stop_patience 12 --early_stop_min_delta 0.002 \
  --freeze_encoder_epochs 15 \
  --phase1_head_lr 5e-4 --phase2_head_lr 2e-4 --phase2_encoder_lr 1e-5 \
  --epochs 120 --batch_size 1 --patch_size 24 --precision 16-mixed \
  --num_devices 1 --num_workers 8 --new_version
```

Notes:
- Task 1/3: ignore segmentation masks. Task 2 (segmentation) will use masks and different heads/metrics (DSC, NSD). The fusion encoder and flags remain consistent.

### Task 3 (Brain Age Regression) – Fusion Finetune
- Canonical modalities: `("T1", "T2")`
- Mapping: `T1→t1`, `T2→t2`
- Label: `age` (single float/int per subject)
- Data dir example: `/mnt/.../fomo-finetune/fomo-task3/Task003_FOMO3_fusion/<subject>/{T1.npy,T2.npy,mask.json,label.txt}`

- Run:
  ```bash
  PY=/mnt/cvlab/scratch/cvlab/home/hantzhan/anaconda3/envs/fomo/bin/python
  cd /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main

  PYTHONPATH=src "$PY" src/finetune.py \
    --taskid 3 \
    --data_dir /mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune/fomo-task3 \
    --save_dir ./runs \
    --model_name unet_xl \
    --fusion_mode auto \
    --modality_mapping "T1=t1,T2=t2" \
    --t1_ckpt /abs/path/to/t1.ckpt \
    --t2_ckpt /abs/path/to/t2.ckpt \
    --epochs 200 --batch_size 2 --num_devices 1 --num_workers 8 --new_version
  ```
- Metrics: MAE, MSE, R2, Pearson correlation are logged by default.
  - Precision: for maximal stability, prefer full precision: `--precision 32-true`.

### Task 2 (Meningioma Segmentation) – Fusion Finetune
- Canonical modalities: `("DWI", "T2FLAIR", "SWI_OR_T2STAR")`
- Label: binary tumor mask per subject (`mask.nii.gz`)
- Config: `task2_config` now sets `task_type="segmentation"`, `num_classes=2`, `label_extension=".nii.gz"`.
- Dataset: unified `src/data/dataset_fusion.py` (`FusionDataset`) handles classification, regression, and segmentation.
- Preprocessing: `src/data/preprocess/fomo2_fusion.py` builds per-subject folders at `<data_dir>/Task002_FOMO2_fusion/<FOMO2_...>/{DWI.npy,T2FLAIR.npy,SWI_OR_T2STAR.npy,mask.json,mask.nii.gz}` from raw `preprocessed/` and `labels/seg.nii.gz`.
- Metrics: Dice and F1 reported by `SupervisedSegModel`. For evaluation exports, `src/evaluator.py` supports surface metrics (Average Surface Distance). NSD can be integrated via yucca surface metrics if required by enabling surface eval.
- Example:
```bash
PY=/path/to/python
cd /path/to/repo
PYTHONPATH=src "$PY" src/finetune.py \
  --taskid 2 \
  --data_dir /path/to/fomo-finetune \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --epochs 120 --batch_size 1 --patch_size 24 --precision 16-mixed \
  --num_devices 1 --num_workers 8 --new_version
```

## Tests
- Fusion correctness: `src/tests/test_fusion_masked_mean.py`
- Weight remap: `src/tests/test_weight_key_remap.py`
- Dataset collation: `src/tests/test_dataset_collation.py`
- E2E finetune forward: `src/tests/test_finetune_multiencoder.py`

- Run:
  ```bash
  PYTHONPATH=src python -m unittest -v \
    src/tests/test_fusion_masked_mean.py \
    src/tests/test_weight_key_remap.py \
    src/tests/test_dataset_collation.py \
    src/tests/test_finetune_multiencoder.py
  ```

## Troubleshooting
- Wrong interpreter (base Python) → `torch._C` error:
  - Use absolute env Python (`.../envs/fomo/bin/python`) or:
    ```bash
    conda run -n fomo bash -lc 'cd <repo> && PYTHONPATH=src python src/finetune.py ...'
    ```
- Lightning import:
  - Install `lightning==2.5.2`, or switch imports to `pytorch_lightning` consistently.
- `InstanceNorm3d` failure:
  - Use larger patch sizes (e.g., `--patch_size 32`), divisible by 8.
- Partial weight-transfer warnings:
  - Expected; new fusion/head layers initialize from scratch; missing groups are allowed.

## Extensibility
- Add a new pretrain group:
  - Extend `global_vocab`; `MaskedMeanFusion.gamma` automatically sizes to `len(global_vocab)`.
- Custom fusion:
  - Implement a new fusion module and plug into `MultiModalEncoderWithFusion`.
- Freeze per-modality encoders:
  - Add freezing logic in `BaseSupervisedModel` or after model construction.
- Dataset variants:
  - Keep per-modality `.npy` + `mask.json` conventions; tests cover core behavior.

## Reference Files
- Pretrain: `src/pretrain.py`
- Finetune: `src/finetune.py`
- Fusion layer: `src/models/fusion/masked_mean.py`
- Multi-encoder: `src/models/multiencoder.py`
- UNet: `src/models/networks/unet.py`
- Heads: `src/models/networks/heads.py`
- Fusion dataset: `src/data/dataset_fusion.py`
- Preprocessing (fusion): `src/data/preprocess/fomo1_fusion.py`
- Task config: `src/data/task_configs.py`


## Finetune implementation status (Task 1)

### Highlights
- **Data**
  - Fusion-style preprocessing: `Task001_FOMO1_fusion/<subject>/{DWI.npy,ADC.npy,T2FLAIR.npy,SWI_OR_T2STAR.npy,mask.json,label.txt}`.
  - `FusionCLSDataset` loads per-subject folders, stacks modalities, supports Yucca DataModule args.
  - Fusion selection via `--fusion_mode {fusion, stacked, auto}`; in fusion mode, splits are built from subject folders.

- **Model**
  - Multi-encoder wrapper with Masked Mean Fusion across scales; per-global-group `gamma` sized to `len(global_vocab)`.
  - Default `unet_xl`; heads use `nn.LazyLinear`.
  - Mapping (FOMO1 → global groups): DWI/ADC→dwi, T2FLAIR→flair, SWI_OR_T2STAR→other.

- **Weights**
  - Fusion: per-group ckpts `--t1_ckpt, --t2_ckpt, --flair_ckpt, --dwi_ckpt, --other_ckpt` or `--modality_ckpts` (no `all`). Missing groups init randomly; no fallback.
  - Stacked: single `--all_ckpt` or `--pretrained_weights_path`.

- **Training/runtime**
  - Subject-level AUROC: aggregates logits by subject and logs `val/auroc_subject`.
  - Subject-level export: JSON + CSV each validation epoch with `--export_subject_probs`.
  - CUDA safety: `--precision 16-mixed`, small `--batch_size` (1–2), `--patch_size` 24–32.
  - Fusion-aware, deterministic split discovery; K-fold controls `--k_folds`, `--fold_index`.
  - Early stopping/ModelCheckpoint monitor `val/auroc_subject` (binary task) with patience/min_delta.
  - Two-phase schedule: freeze encoders (head LR 5e-4) then unfreeze (encoder LR 1e-5, head LR 2e-4).
  - Stability options: grad accumulation/clipping, channels-last, SWA, EMA (`--use_ema`).

- **Docs/tests**
  - This guide updated; unit tests cover fusion math, weight remap, dataset collation, multi-encoder forward, and subject-level AUROC aggregation.

### Current
- Fusion finetune path implemented end-to-end:
  - Fusion dataset and preprocessing; `FusionCLSDataset` integrated with Yucca DataModule.
  - `--fusion_mode` switch (fusion/stacked/auto); per-group ckpt flags; no `all` fallback in fusion.
  - Stratified subject-level K-fold: `--k_folds`, `--fold_index`.
  - Subject-level AUROC (`val/auroc_subject`) with EarlyStopping and best-checkpoint monitoring.
  - Two-phase finetune (freeze encoders, then unfreeze with separate LRs).
- EMA enabled and saved: validation swaps to EMA; best EMA checkpoint saved as `ema-best.ckpt`.
- Ensembling support:
  - Utilities and CLI for subject-wise ensembling; `tools/summarize_kfold.py` to report per-fold and ensembled AUROC.
- Calibration support: `TemperatureScaler` utilities.
- Subject-level export and CSV; optional validation TTA flag.


