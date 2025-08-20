#!/usr/bin/env python

import argparse
import os
import logging
import torch
import lightning as L
# import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from models.supervised_base import BaseSupervisedModel
from augmentations.finetune_augmentation_presets import (
    get_finetune_augmentation_params,
)
from augmentations.anisotropic_augmentation import (
    get_anisotropic_augmentation_for_task,
    AnisotropicSpatialTransform,
)
from utils.utils import (
    SimplePathConfig,
    setup_seed,
    find_checkpoint,
    load_pretrained_weights,
)
from utils.enhanced_callbacks import (
    EnhancedModelCheckpoint,
    LossPlottingCallback,
    MetricsStabilityCallback,
    SmoothLoggingCallback,
)

from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
)

from yucca.modules.data.augmentation.YuccaAugmentationComposer import (
    YuccaAugmentationComposer,
)
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from data.cls_datamodule import CLSDataModule
from yucca.modules.callbacks.loggers import YuccaLogger

from yucca.pipeline.configuration.split_data import get_split_config
from yucca.pipeline.configuration.configure_paths import detect_version
from data.dataset import CLSDataset
from data.dataset_fusion import FusionCLSDataset
from data.task_configs import task1_config, task2_config, task3_config, hbn_config
from torch.utils.data import SequentialSampler


def get_task_config(taskid):
    if taskid == 1:
        task_cfg = task1_config
    elif taskid == 2:
        task_cfg = task2_config
    elif taskid == 3:
        task_cfg = task3_config
    elif taskid == 4:
        task_cfg = hbn_config
    else:
        raise ValueError(f"Unknown taskid: {taskid}. Supported IDs are 1, 2, and 3")

    return task_cfg


def main():
    # Quiet by default; raise to INFO/DEBUG via your launcher if needed
    logging.getLogger().setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to data directory",
        default="./data/preprocessed",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to save models and results",
        default="./data/models",
    )
    parser.add_argument(
        "--pretrained_weights_path", type=str, help="Ckpt to finetune", default=None
    )
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet_xl",
        help="Model name defined in models.networks (unet_b, unet_xl, etc.)",
    )
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--patch_size", type=str, default="32", 
                        help="Patch size: single int for isotropic (32) or comma-separated for anisotropic (96,96,24)")
    parser.add_argument("--starting_filters", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default=None)
    # Multi-encoder options
    parser.add_argument("--use_multi_encoder", action="store_true")
    parser.add_argument(
        "--modality_ckpts",
        type=str,
        default=None,
        help="Comma-separated mapping modalityGroup=abs_path for pretrain ckpts. Keys in {t1,t2,flair,dwi,other,all}",
    )
    # Explicit per-group checkpoints (preferred)
    parser.add_argument("--t1_ckpt", type=str, default=None)
    parser.add_argument("--t2_ckpt", type=str, default=None)
    parser.add_argument("--flair_ckpt", type=str, default=None)
    parser.add_argument("--dwi_ckpt", type=str, default=None)
    parser.add_argument("--other_ckpt", type=str, default=None)
    # Single combined checkpoint for legacy stacked finetune
    parser.add_argument("--all_ckpt", type=str, default=None)
    parser.add_argument(
        "--modality_mapping",
        type=str,
        default=None,
        help="Comma-separated mapping finetuneMod=pretrainGroup, e.g. DWI=dwi,ADC=dwi,T2FLAIR=flair,SWI_OR_T2STAR=other",
    )
    # Modality enable/disable switches
    parser.add_argument(
        "--enabled_modalities",
        type=str,
        default=None,
        help="Comma-separated list of modalities to enable (e.g., 'DWI,ADC'). If not specified, all modalities are enabled.",
    )
    parser.add_argument(
        "--disabled_modalities", 
        type=str,
        default=None,
        help="Comma-separated list of modalities to disable. Mutually exclusive with --enabled_modalities.",
    )
    # Fusion mechanism options
    parser.add_argument(
        "--fusion_type",
        type=str,
        choices=[
            "masked_mean", "attention", "channel_attention", "spatial_attention", "hybrid_attention",
            "learnable_weighted", "channel_gated", "uncertainty_weighted"
        ],
        default="masked_mean",
        help="Type of fusion mechanism to use between modalities",
    )
    parser.add_argument("--allow_missing_modalities", action="store_true")
    parser.add_argument("--modality_dropout_p_start", type=float, default=0.2)
    parser.add_argument("--modality_dropout_p_end", type=float, default=0.5)
    parser.add_argument("--modality_dropout_warmup_epochs", type=int, default=50)
    # Hardware configuration
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fast_dev_run", action="store_true")
    # Dataset input mode switch
    parser.add_argument(
        "--fusion_mode",
        type=str,
        choices=["auto", "fusion", "stacked"],
        default="auto",
        help="Dataset format: fusion=per-modality folders, stacked=single .npy, auto=detect",
    )
    # Experiment tracking
    parser.add_argument("--new_version", action="store_true")
    parser.add_argument(
        "--augmentation_preset",
        type=str,
        choices=["all", "basic", "anisotropic", "none"],
        default="basic",
    )
    # Training Parameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--smoothing_window", type=int, default=10, help="Window size for loss smoothing (1 to disable)")
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--train_batches_per_epoch", type=int, default=100)
    # Task Configuration
    parser.add_argument(
        "--taskid",
        type=int,
        required=True,
        help="Task ID (1: FOMO1 classification, 2: FOMO2 classification, 3: FOMO3 regression)",
    )
    # K-Fold parameters
    parser.add_argument("--k_folds", type=int, default=1, help="Total number of folds (K). If >1, use K-fold.")
    parser.add_argument("--fold_index", type=int, default=0, help="Current fold index (0-based).")
    # Class balance
    parser.add_argument("--balance_train", type=str, default="none", choices=["none", "match_upsample"],
                        help="Class balancing strategy for fusion mode training list")
    # Early stopping & schedule
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.002)
    parser.add_argument("--disable_early_stop", action="store_true")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=15)
    parser.add_argument("--phase1_head_lr", type=float, default=5e-4)
    parser.add_argument("--phase2_head_lr", type=float, default=2e-4)
    parser.add_argument("--phase2_encoder_lr", type=float, default=1e-5)
    # Gradient clipping
    parser.add_argument("--grad_clip_val", type=float, default=0.0)
    parser.add_argument(
        "--grad_clip_algo",
        type=str,
        default="norm",
        choices=["norm", "value"],
        help="Gradient clipping algorithm (norm or value)",
    )
    # Classification head regularization
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--cls_head_dropout_p", type=float, default=0.0)
    
    # Regression-specific parameters for brain age prediction
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "mae", "huber"],
                        help="Loss function for regression (mae recommended for brain age)")
    parser.add_argument("--age_normalization", action="store_true", default=False,
                        help="Normalize age values for stable training (recommended)")
    parser.add_argument("--age_mean", type=float, default=50.0,
                        help="Mean age for normalization (default: 50 years)")
    parser.add_argument("--age_std", type=float, default=15.0,
                        help="Age standard deviation for normalization (default: 15 years)")
    
    # Validation/Test-time augmentation (multi-view averaging)
    parser.add_argument("--val_tta_enable", action="store_true")
    parser.add_argument("--val_tta_views", type=int, default=8,
                        help="Number of deterministic flip views to average (max 8)")
    parser.add_argument("--val_tta_offsets", type=int, default=1,
                        help="Number of deterministic translation offsets (center + axis shifts), max 7")
    parser.add_argument("--val_tta_offset_frac", type=float, default=0.25,
                        help="Offset as fraction of patch size along each axis (0..0.5)")
    parser.add_argument("--val_tta_batch_size", type=int, default=8,
                        help="Micro-batch size for TTA combos per validation step to fully utilize GPU")
    # LR scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "plateau"])
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=4)
    parser.add_argument("--plateau_threshold", type=float, default=1e-3)
    parser.add_argument("--plateau_cooldown", type=int, default=0)
    parser.add_argument("--plateau_min_lr", type=float, default=1e-7)
    # Split Configuration
    parser.add_argument("--split_method", type=str, default="simple_train_val_split")
    parser.add_argument("--split_param", type=str, help="Split parameter", default=0.2)
    parser.add_argument(
        "--split_idx", type=int, default=0, help="Index of the split to use for kfold"
    )
    parser.add_argument(
        "--experiment", type=str, default="experiment", help="name of experiment"
    )
    args = parser.parse_args()

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
    
    # Validate all dimensions are divisible by 8
    for i, dim in enumerate(patch_size_tuple):
        assert dim % 8 == 0, f"Patch size dimension {i} must be divisible by 8, got {dim}"

    # Set up task configuration
    task_cfg = get_task_config(args.taskid)
    task_type = task_cfg["task_type"]
    task_name = task_cfg["task_name"]
    num_classes = task_cfg["num_classes"]
    modalities = len(task_cfg["modalities"])
    labels = task_cfg["labels"]
    
    # Use task-specific patch size if not explicitly provided or if using defaults
    if args.patch_size == "32":  # Default value
        task_patch_size = task_cfg.get("patch_size", None)
        if task_patch_size is not None:
            patch_size_tuple = tuple(task_patch_size)
            print(f"📐 Using task-specific anisotropic patch size: {patch_size_tuple}")
        else:
            print(f"📐 Using default isotropic patch size: {patch_size_tuple}")
    else:
        print(f"📐 Using user-specified patch size: {patch_size_tuple}")

    # Determine run type: we consider fusion pretrain ckpts as 'finetune'
    has_any_group_ckpt = any(
        getattr(args, k) is not None and str(getattr(args, k)).strip() != ""
        for k in ["t1_ckpt", "t2_ckpt", "flair_ckpt", "dwi_ckpt", "other_ckpt"]
    )
    run_type = "finetune" if (args.pretrained_weights_path is not None or has_any_group_ckpt or args.all_ckpt is not None) else "from_scratch"
    # Defer weight transfer decision until after fusion detection
    do_weight_transfer = False
    experiment_name = f"{run_type}_{args.experiment}_{args.taskid}"

    logging.info(f"Using num_workers: {args.num_workers}, num_devices: {args.num_devices}")
    logging.info(f"Task type: {task_type}")
    logging.debug(f"ARGS: {args}")

    # Set up directory structure
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, task_name)
    # If fusion dataset requested/exists, use it consistently for split discovery and loading
    fusion_dir = os.path.join(data_dir, f"{task_name}_fusion")
    if args.fusion_mode == "fusion":
        use_fusion = True
    elif args.fusion_mode == "stacked":
        use_fusion = False
    else:
        use_fusion = os.path.isdir(fusion_dir)
    train_data_dir_effective = fusion_dir if use_fusion else train_data_dir

    # Configure weight transfer intent based on user mode
    if use_fusion:
        do_weight_transfer = (
            (args.modality_ckpts is not None) or
            any(getattr(args, k) is not None for k in ["t1_ckpt", "t2_ckpt", "flair_ckpt", "dwi_ckpt", "other_ckpt"]) 
        )
    else:
        do_weight_transfer = (args.all_ckpt is not None) or (args.pretrained_weights_path is not None)

    # Path where logs, checkpoints etc is stored
    save_dir = os.path.join(args.save_dir, task_name, args.model_name)

    # Handle versioning for experiment tracking with meaningful names
    continue_from_most_recent = not args.new_version
    
    # Create experiment-specific directory structure
    experiment_name = args.experiment if args.experiment else "default_experiment"
    
    # Clean experiment name for filesystem compatibility
    import re
    clean_experiment_name = re.sub(r'[^\w\-_\.]', '_', experiment_name)
    
    # Build hierarchical directory: save_dir/experiment_name/version_X
    experiment_dir = os.path.join(save_dir, clean_experiment_name)
    
    if continue_from_most_recent and os.path.exists(experiment_dir):
        # Continue existing experiment
        version = detect_version(experiment_dir, continue_from_most_recent)
        version_dir = os.path.join(experiment_dir, f"version_{version}")
    else:
        # New experiment or forced new version
        if os.path.exists(experiment_dir):
            # Experiment exists, create new version
            version = detect_version(experiment_dir, False)  # Force new version
        else:
            # First run of this experiment
            version = 0
        version_dir = os.path.join(experiment_dir, f"version_{version}")
    
    ensure_dir_exists(version_dir)
    
    # Log the experiment structure
    logging.info(f"Experiment dir: {version_dir} (base={save_dir}, name={clean_experiment_name}, version={version})")

    # Create dataset splits
    if args.split_method == "kfold":
        split_param = int(args.split_param)
    elif args.split_method == "simple_train_val_split":
        split_param = float(args.split_param)
    else:
        split_param = args.split_param

    if use_fusion:
        # Build splits from subject directories (full paths) inside fusion_dir
        subject_dirs = [
            os.path.join(fusion_dir, d)
            for d in sorted(os.listdir(fusion_dir))
            if os.path.isdir(os.path.join(fusion_dir, d))
        ]
        assert len(subject_dirs) > 0, f"No subject directories found in fusion dir: {fusion_dir}"

        # Read labels for stratification (except for segmentation tasks)
        if task_type == "segmentation":
            # For segmentation, we don't need label-based stratification
            # since the "labels" are segmentation masks, not discrete classes
            labels = None
        else:
            def read_label(subj_dir: str) -> float:
                path = os.path.join(subj_dir, "label.txt")
                return float(open(path, "r").read().strip())

            labels = [read_label(sd) for sd in subject_dirs]

        import random
        rnd = random.Random(2025)

        # Task-type aware splitting strategy
        if task_type == "segmentation":
            # For segmentation: simple random split (no stratification needed)
            rnd.shuffle(subject_dirs)
            pos = subject_dirs  # Use all data as "positive" for unified logic below
            neg = []  # Empty negative class for segmentation
        elif task_type == "regression":
            # For regression: simple random split (no stratification needed for continuous labels)
            combined = list(zip(subject_dirs, labels))
            rnd.shuffle(combined)
            subject_dirs_shuffled = [sd for sd, _ in combined]
            pos = subject_dirs_shuffled  # Use all data as "positive" for unified logic below
            neg = []  # Empty negative class for regression
        else:
            # For classification: stratified split by binary classes
            pos = [sd for sd, y in zip(subject_dirs, labels) if int(y) == 1]
            neg = [sd for sd, y in zip(subject_dirs, labels) if int(y) == 0]
            rnd.shuffle(pos)
            rnd.shuffle(neg)

        def make_kfolds(items: list, k: int) -> list[list]:
            return [items[i::k] for i in range(k)] if k > 0 else [items]

        k = max(1, int(args.k_folds))
        f = max(0, int(args.fold_index))
        if k > 1:
            pos_folds = make_kfolds(pos, k)
            neg_folds = make_kfolds(neg, k)
            f = min(f, k - 1)
            val_list = pos_folds[f] + neg_folds[f]
            train_list = [x for i in range(k) if i != f for x in (pos_folds[i] + neg_folds[i])]
        else:
            # Simple split fallback
            all_items = pos + neg
            rnd.shuffle(all_items)
            if args.split_method == "simple_train_val_split":
                frac = float(split_param)
                n_train = max(1, int(len(all_items) * (1 - frac)))
                train_list = all_items[:n_train]
                val_list = all_items[n_train:]
            else:
                n_train = max(1, int(len(all_items) * 0.8))
                train_list = all_items[:n_train]
                val_list = all_items[n_train:]

        # Optional balancing (fusion mode only)
        if str(args.balance_train) == "match_upsample":
            # Determine class labels for train_list
            def read_label(subj_dir: str) -> int:
                path = os.path.join(subj_dir, "label.txt")
                return int(float(open(path, "r").read().strip()))
            pos_train = [sd for sd in train_list if read_label(sd) == 1]
            neg_train = [sd for sd in train_list if read_label(sd) == 0]
            if len(pos_train) > 0 and len(neg_train) > 0:
                if len(pos_train) < len(neg_train):
                    # upsample positives
                    reps = (len(neg_train) + len(pos_train) - 1) // max(1, len(pos_train))
                    train_list = (pos_train * reps + neg_train)[: len(neg_train) * 2]
                elif len(neg_train) < len(pos_train):
                    reps = (len(pos_train) + len(neg_train) - 1) // max(1, len(neg_train))
                    train_list = (neg_train * reps + pos_train)[: len(pos_train) * 2]
                rnd.shuffle(train_list)

        class SimpleSplitsConfig:
            def __init__(self, train_list, val_list):
                self._train = train_list
                self._val = val_list
            def train(self, split_idx):
                return self._train
            def val(self, split_idx):
                return self._val

        splits_config = SimpleSplitsConfig(train_list, val_list)
    else:
        path_config = SimplePathConfig(train_data_dir=train_data_dir_effective)
        splits_config = get_split_config(
            method=args.split_method,
            param=split_param,
            path_config=path_config,
        )

    # Set up seed for reproducability
    seed = setup_seed(continue_from_most_recent)
    # Look for existing checkpoint if continuing training
    ckpt_path = find_checkpoint(version_dir, continue_from_most_recent)

    # Calculate training metrics
    effective_batch_size = args.num_devices * args.batch_size
    train_dataset_size = len(splits_config.train(args.split_idx))
    val_dataset_size = len(splits_config.val(args.split_idx))
    max_iterations = int(args.epochs * args.train_batches_per_epoch)

    # the config contains all the parameters needed for training and is used by lightning module, data module, and trainer
    config = {
        # Task information
        "task": task_name,
        "task_id": args.taskid,
        "task_type": task_type,
        "experiment": experiment_name,
        "model_name": args.model_name,
        "model_dimensions": "3D",
        "run_type": run_type,
        
        # Split configuration
        "split_method": args.split_method,
        "split_param": split_param,
        "split_idx": args.split_idx,
        
        # Directories
        "save_dir": save_dir,
        "train_data_dir": train_data_dir,
        "version_dir": version_dir,
        "version": version,
        
        # Checkpoint
        "ckpt_path": ckpt_path,
        "pretrained_weights_path": args.pretrained_weights_path,
        
        # Reproducibility
        "seed": seed,
        
        # Dataset properties
        "num_classes": num_classes,
        "num_modalities": modalities,
        "image_extension": ".npy",
        "allow_missing_modalities": False,
        "labels": labels,
        
        # Training parameters
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "learning_rate": args.learning_rate,
        "patch_size": patch_size_tuple,  # Use parsed anisotropic patch size
        "target_spacing": task_cfg.get("target_spacing", [1.0, 1.0, 1.0]),  # Add spacing config
        "starting_filters": int(args.starting_filters),
        "precision": args.precision,
        "augmentation_preset": args.augmentation_preset,
        "epochs": args.epochs,
        "train_batches_per_epoch": args.train_batches_per_epoch,
        "effective_batch_size": effective_batch_size,

		# Optimizer schedule (enable lower LR for encoders and optional freeze)
		"freeze_encoder_epochs": args.freeze_encoder_epochs,
		"phase1_head_lr": args.phase1_head_lr,
		"phase2_head_lr": args.phase2_head_lr,
		"phase2_encoder_lr": args.phase2_encoder_lr,
        
        # Dataset metrics
        "train_dataset_size": train_dataset_size,
        "val_dataset_size": val_dataset_size,
        "max_iterations": max_iterations,
        
        # Hardware settings
        "num_devices": args.num_devices,
        "num_workers": args.num_workers,
        
        # Model compilation
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        # Multi-encoder config
        "use_multi_encoder": use_fusion,

		# Validation/Test-time augmentation
		"val_tta_enable": args.val_tta_enable,
		"val_tta_views": int(max(1, min(8, args.val_tta_views))),
		"val_tta_offsets": int(max(1, min(7, args.val_tta_offsets))),
		"val_tta_offset_frac": float(max(0.0, min(0.5, args.val_tta_offset_frac))),
		"val_tta_batch_size": int(max(1, args.val_tta_batch_size)),

		# Head regularization
		"label_smoothing": max(0.0, min(0.3, float(args.label_smoothing))),
		"cls_head_dropout_p": max(0.0, min(0.8, float(args.cls_head_dropout_p))),

		# Regression-specific parameters for brain age
		"loss_type": str(args.loss_type),
		"age_normalization": bool(args.age_normalization),
		"age_mean": float(args.age_mean),
		"age_std": float(args.age_std),

		# LR scheduler
		"lr_scheduler": str(args.lr_scheduler),
		"plateau_factor": float(args.plateau_factor),
		"plateau_patience": int(args.plateau_patience),
		"plateau_threshold": float(args.plateau_threshold),
		"plateau_cooldown": int(args.plateau_cooldown),
		"plateau_min_lr": float(args.plateau_min_lr),

		# Gradient clipping (for tracking)
		"grad_clip_val": float(args.grad_clip_val),
		"grad_clip_algorithm": str(args.grad_clip_algo),
        
        # Trainer specific params
        "fast_dev_run": args.fast_dev_run,
    }

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

    # Enhanced checkpoint callback with terminal feedback
    checkpoint_callback = EnhancedModelCheckpoint(
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
    )
    
    # Loss plotting callback that shares log_every_n_steps parameter
    loss_plotting_callback = LossPlottingCallback(
        save_dir=version_dir,
        log_every_n_steps=args.log_every_n_steps,
        plot_every_n_epochs=1,  # Save plots every epoch
        smoothing_window=args.smoothing_window,     # Window size for moving average smoothing
        best_metrics_file=os.path.join(version_dir, "best_metrics.txt")  # Pass best metrics file path
    )
    
    # Metrics stability callback for better monitoring
    if task_type == "classification" and num_classes == 2:
        # Use AUROC monitoring for binary classification
        metrics_stability_callback = MetricsStabilityCallback(
            window_size=5, 
            primary_metric="val/auroc_subject"
        )
    else:
        # Use accuracy monitoring for classification, correlation for regression
        if task_type == "classification":
            primary_metric = "val/accuracy"
        elif task_type == "regression":
            # Align with checkpoint monitor and logged key from model
            primary_metric = "val/corr"
        else:
            primary_metric = "val/loss"  # Fallback
            
        metrics_stability_callback = MetricsStabilityCallback(
            window_size=5, 
            primary_metric=primary_metric
        )
    
    # Smooth logging callback for console output
    smooth_logging_callback = SmoothLoggingCallback(
        smoothing_window=args.smoothing_window
    )
    
    callbacks = [checkpoint_callback, loss_plotting_callback, metrics_stability_callback, smooth_logging_callback]
    
    if not args.disable_early_stop:
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
        )
        callbacks.append(early_stop)

    # Create logger for metrics
    yucca_logger = YuccaLogger(
        save_dir=experiment_dir,
        version=version,
        steps_per_epoch=args.train_batches_per_epoch,
    )
    loggers = [yucca_logger]

    # Display quality enhancement features
    logging.debug("Quality features: checkpoint feedback, loss plotting enabled")
    if task_type == "classification" and num_classes == 2:
        logging.debug("AUROC stability monitoring for binary classification")
        logging.debug(f"Best checkpoints monitor: {monitor_metric}")
    elif task_type == "classification":
        logging.debug("Accuracy stability monitoring for multi-class classification")
        logging.debug(f"Best checkpoints monitor: {monitor_metric}")
    elif task_type == "regression":
        logging.debug("Pearson correlation monitoring (val/corr) for brain age regression")
        logging.debug(f"Best checkpoints monitor: {monitor_metric}")
        logging.debug("Enhanced metrics: MAE and Pearson Correlation")
    else:
        logging.debug("Loss stability monitoring for other tasks")
        logging.debug(f"Best checkpoints monitor: {monitor_metric}")
    logging.debug(f"Training progress plots saved to: {version_dir}")
    logging.debug(f"Metrics logged every {args.log_every_n_steps} steps")


    # Configure augmentations based on preset
    if args.augmentation_preset == "anisotropic":
        # Use anisotropic augmentation system for non-isotropic data
        logging.info(f"Using anisotropic augmentations for task: {task_name}")
        
        # Extract task number from task_name (e.g., "Task001_FOMO1" -> 1)
        task_id = int(task_name.split('_')[-1][-1])  # Get last character and convert to int
        
        anisotropic_augmenter = get_anisotropic_augmentation_for_task(
            task_id=task_id,
            patch_size=config["patch_size"],
            spacing=config.get("target_spacing")
        )
        
        train_transforms = anisotropic_augmenter.train_transforms
        val_transforms = anisotropic_augmenter.val_transforms
    else:
        # Use yucca augmentation system for standard training
        aug_params = get_finetune_augmentation_params(args.augmentation_preset)
        
        # Map task type for augmentation composer (regression uses same augmentations as classification)
        aug_task_type = "classification" if task_type in ["classification", "regression"] else task_type
        
        augmenter = YuccaAugmentationComposer(
            patch_size=config["patch_size"],
            task_type_preset=aug_task_type,
            parameter_dict=aug_params,
            deep_supervision=False,
        )
        train_transforms = augmenter.train_transforms
        val_transforms = augmenter.val_transforms

    # Create the data module that handles loading and batching
    # Choose dataset class: FusionCLSDataset if folder-style fusion preprocessing detected
    dataset_cls = FusionCLSDataset if use_fusion else CLSDataset

    data_module = YuccaDataModule(
        train_dataset_class=dataset_cls,  # Supports both stacked and per-modality fusion
        composed_train_transforms=train_transforms,
        composed_val_transforms=val_transforms,
        patch_size=config["patch_size"],
        batch_size=config["batch_size"],
        train_data_dir=(train_data_dir if dataset_cls is CLSDataset else fusion_dir),
        image_extension=config["image_extension"],
        task_type=config["task_type"],
        splits_config=splits_config,
        split_idx=config["split_idx"],
        num_workers=args.num_workers,
        val_sampler=SequentialSampler,
    )
    # If the YuccaDataModule exposes val_batch_size, set it when available
    if hasattr(data_module, 'val_batch_size'):
        try:
            data_module.val_batch_size = max(1, int(config.get("val_batch_size", config["batch_size"])))
        except Exception:
            pass
    # Print dataset information
    logging.info(f"Train/Val sizes: {train_dataset_size}/{val_dataset_size}; run_type={run_type}")
    logging.info(
        f"Start training: {max_iterations} iters, {args.epochs} epochs, effective_batch={effective_batch_size}"
    )

    # Initialize wandb logging
    # wandb.init(
    #     project="fomo-finetuning",
    #     name=f"{config['experiment']}_version_{config['version']}",
    # )

    # # Create wandb logger for Lightning
    # wandb_logger = L.pytorch.loggers.WandbLogger(
    #     project="fomo-finetuning",
    #     name=f"{config['experiment']}_version_{config['version']}",
    #     log_model=True,
    # )
    # loggers.append(wandb_logger)

    # Create model and trainer
    # If fusion mode, include modality names and group mapping in config to build the wrapper
    if use_fusion:
        finetune_modalities = list(task_cfg["modalities"])  # e.g., FOMO1
        
        # Handle modality enable/disable switches
        enabled_modalities = None
        if args.enabled_modalities is not None and args.disabled_modalities is not None:
            raise ValueError("Cannot specify both --enabled_modalities and --disabled_modalities")
        
        if args.enabled_modalities is not None:
            enabled_modalities = [m.strip() for m in args.enabled_modalities.split(",") if m.strip()]
            # Validate enabled modalities
            invalid = set(enabled_modalities) - set(finetune_modalities)
            if invalid:
                raise ValueError(f"Invalid enabled modalities: {invalid}. Available: {finetune_modalities}")
        
        if args.disabled_modalities is not None:
            disabled_modalities = [m.strip() for m in args.disabled_modalities.split(",") if m.strip()]
            # Validate disabled modalities
            invalid = set(disabled_modalities) - set(finetune_modalities)
            if invalid:
                raise ValueError(f"Invalid disabled modalities: {invalid}. Available: {finetune_modalities}")
            enabled_modalities = [m for m in finetune_modalities if m not in disabled_modalities]
            if len(enabled_modalities) == 0:
                raise ValueError("Cannot disable all modalities")
        
        config["multi_encoder_modalities"] = finetune_modalities
        config["enabled_modalities"] = enabled_modalities  # None means all enabled
        config["fusion_type"] = args.fusion_type
        
        logging.debug(f"Modality configuration")
        logging.debug(f"Available modalities: {finetune_modalities}")
        if enabled_modalities is not None:
            logging.debug(f"Enabled modalities: {enabled_modalities}")
            logging.debug(f"Disabled modalities: {[m for m in finetune_modalities if m not in enabled_modalities]}")
        else:
            logging.debug("All modalities enabled by default")
        logging.debug(f"Fusion type: {args.fusion_type}")
        
        default_mapping = {
            # FOMO1 canonical
            "DWI": "dwi",
            "ADC": "dwi",
            "T2FLAIR": "flair",
            "SWI_OR_T2STAR": "other",
            # FOMO3 canonical
            "T1": "t1",
            "T2": "t2",
        }
        modality_to_global_group = default_mapping.copy()
        if args.modality_mapping is not None:
            for kv in args.modality_mapping.split(","):
                if kv.strip() == "":
                    continue
                k, v = kv.split("=")
                modality_to_global_group[k.strip()] = v.strip()
        config["modality_to_global_group"] = modality_to_global_group
        config["global_vocab"] = ["t1","t2","flair","dwi","other"]
    else:
        if args.use_multi_encoder:
            logging.warning("--use_multi_encoder ignored in stacked mode; using single encoder.")

    model = BaseSupervisedModel.create(
        task_type=task_type,
        config=config,
        learning_rate=args.learning_rate,
        do_compile=args.compile,
        compile_mode="default" if args.compile_mode is None else args.compile_mode,
    )

    # Display accurate parameter counts considering freeze strategy
    logging.info("Model Parameter Summary")
    if hasattr(model, 'get_effective_parameter_counts'):
        counts = model.get_effective_parameter_counts()
        if counts['freeze_active']:
            logging.info(f"FROZEN encoder params: {counts['frozen']:,} (LR=0)")
            logging.info(f"TRAINING head params: {counts['trainable']:,}")
            logging.info(f"Total Params: {counts['total']:,} (~{counts['total'] * 4 / 1024**2:.1f} MB fp32)")
            logging.debug(
                f"Note: Lightning reports all as trainable (AMP), but {counts['frozen']:,} have LR=0"
            )
        else:
            logging.info(f"All TRAINABLE: {counts['trainable']:,} params (~{counts['total'] * 4 / 1024**2:.1f} MB fp32)")

    # Create Lightning trainer
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=loggers,
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        num_nodes=1,
        devices=args.num_devices,
        default_root_dir=experiment_dir,
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches_per_epoch,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=int(max(1, args.log_every_n_steps)),
        num_sanity_val_steps=int(max(0, args.num_sanity_val_steps)),
        accumulate_grad_batches=int(max(1, args.accumulate_grad_batches)),
        gradient_clip_val=args.grad_clip_val if args.grad_clip_val and args.grad_clip_val > 0 else 0.0,
        gradient_clip_algorithm=args.grad_clip_algo,
    )

    # Load pretrained weights if requested
    if do_weight_transfer:
        logging.info("Transferring pretrained weights where available")
        logging.debug(f"Checkpoint path: {ckpt_path}")
        assert ckpt_path is None, (
            "Error: You're attempting to load pretrained weights while "
            "simultaneously continuing from a checkpoint. This creates "
            "conflicting weight sources. Use either --pretrained_weights_path "
            "for finetuning OR continue training without the --new_version flag, "
            "but not both."
        )
        if not use_fusion:
            # Single-encoder (stacked) path: use --all_ckpt (preferred) or --pretrained_weights_path
            single_ckpt = args.all_ckpt if args.all_ckpt is not None else args.pretrained_weights_path
            state_dict = load_pretrained_weights(single_ckpt, args.compile)
            state_dict = state_dict['state_dict']
            num_successful_weights_transferred = model.load_state_dict(state_dict=state_dict, strict=False)
        else:
            # Multi-encoder: map FOMO1 modalities to pretrain groups and load per-encoder
            # Build ckpt map from explicit flags first; then merge legacy --modality_ckpts (no 'all' fallback)
            ckpt_map = {}
            explicit = {
                "t1": args.t1_ckpt,
                "t2": args.t2_ckpt,
                "flair": args.flair_ckpt,
                "dwi": args.dwi_ckpt,
                "other": args.other_ckpt,
            }
            for k_group, v_path in explicit.items():
                if v_path is not None and str(v_path).strip() != "":
                    ckpt_map[k_group] = v_path
            if args.modality_ckpts is not None:
                for kv in args.modality_ckpts.split(','):
                    if kv.strip() == "":
                        continue
                    k, v = kv.split('=')
                    k = k.strip()
                    if k == 'all':
                        continue  # do not use implicit 'all' fallback
                    if k not in ckpt_map:
                        ckpt_map[k] = v.strip()

            mapping = config.get("modality_to_global_group", {})

            merged_state = {}
            finetune_modalities = config.get("multi_encoder_modalities", [])
            for i, mod in enumerate(finetune_modalities):
                group = mapping.get(mod, None)
                if group is None:
                    logging.warning(f"No global group mapping for modality {mod}. Skipping weight load.")
                    continue
                src_ckpt = ckpt_map.get(group, None)
                if src_ckpt is None:
                    logging.warning(f"No checkpoint provided for group '{group}'. {mod} will use random init.")
                    continue
                sd = load_pretrained_weights(src_ckpt, args.compile)["state_dict"]
                for k, v in sd.items():
                    if not k.startswith("model.encoder."):
                        continue
                    rest = k[len("model.encoder."):]
                    target_key = f"model.encoder.encoders.{mod}.{rest}"
                    merged_state[target_key] = v

            num_successful_weights_transferred = model.load_state_dict(
                state_dict=merged_state, strict=False
            )
        if num_successful_weights_transferred == 0:
            logging.warning("No weights were successfully transferred; proceeding with random init.")
    else:
        logging.info("Training from scratch, no weights will be transferred")

    # Start training
    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)
    # wandb.finish()


if __name__ == "__main__":
    main()
