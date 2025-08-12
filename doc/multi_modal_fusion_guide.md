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
  - Metrics: MAE, MSE, R2 are logged by default. Add Pearson r if desired.
  - Precision: for maximal stability, prefer full precision: `--precision 32-true`.

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


