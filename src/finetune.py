#!/usr/bin/env python

import argparse
import os
import logging
import torch
import lightning as L
# import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from utils.callbacks import ModelEMACallback
try:
    from lightning.pytorch.callbacks import StochasticWeightAveraging
except Exception:
    StochasticWeightAveraging = None

from models.supervised_base import BaseSupervisedModel
from augmentations.finetune_augmentation_presets import (
    get_finetune_augmentation_params,
)
from utils.utils import (
    SimplePathConfig,
    setup_seed,
    find_checkpoint,
    load_pretrained_weights,
)

from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
)

from yucca.modules.data.augmentation.YuccaAugmentationComposer import (
    YuccaAugmentationComposer,
)
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
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
    logging.getLogger().setLevel(logging.INFO)

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
    parser.add_argument("--patch_size", type=int, default=32)
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
        choices=["all", "basic", "none"],
        default="basic",
    )
    # Training Parameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
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
    # Early stopping & schedule
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.002)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=15)
    parser.add_argument("--phase1_head_lr", type=float, default=5e-4)
    parser.add_argument("--phase2_head_lr", type=float, default=2e-4)
    parser.add_argument("--phase2_encoder_lr", type=float, default=1e-5)
    
    # Enhanced training features
    parser.add_argument("--enable_enhanced_aggregation", action="store_true",
                        help="Enable multiple subject-level aggregation methods")
    parser.add_argument("--enable_training_visualization", action="store_true",
                        help="Enable real-time training visualization")
    parser.add_argument("--enhanced_early_stopping", type=str, default="standard",
                        choices=["standard", "robust", "adaptive"],
                        help="Type of early stopping to use")
    parser.add_argument("--visualization_update_freq", type=int, default=1,
                        help="Update visualization every N epochs")
    
    # Optimization/stability options
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Accumulate gradients for this many steps before optimizer step")
    parser.add_argument("--grad_clip_val", type=float, default=0.0,
                        help="If >0, clip gradient norm to this value")
    parser.add_argument("--channels_last", action="store_true",
                        help="Use channels-last memory format for model parameters")
    parser.add_argument("--use_swa", action="store_true",
                        help="Enable Stochastic Weight Averaging (SWA) callback")
    parser.add_argument("--swa_lrs", type=float, default=1e-4,
                        help="SWA learning rate used by the SWA callback if enabled")
    # Subject-level export
    parser.add_argument("--export_subject_probs", action="store_true",
                        help="Export per-subject probabilities/targets as JSON each validation epoch")
    parser.add_argument("--val_tta", action="store_true",
                        help="Enable light test-time augmentation during validation for classification")
    # Scheduling & LR policy (optional; defaults are off or cosine)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "cosine_restarts", "one_cycle", "none"],
                        help="Learning rate scheduler policy")
    parser.add_argument("--one_cycle_max_lr", type=float, default=None,
                        help="Max LR for one-cycle scheduler (defaults to learning_rate if None)")
    parser.add_argument("--apply_layerwise_lr_decay", action="store_true",
                        help="Enable simple layer-wise LR decay on encoder params")
    parser.add_argument("--layerwise_lr_decay_gamma", type=float, default=0.0,
                        help="Decay factor per depth bin (e.g., 0.8)")
    # Loss options
    parser.add_argument("--class_weights", type=str, default=None,
                        help="Comma-separated class weights, e.g., '1.0,2.0' for binary")
    parser.add_argument("--focal_gamma", type=float, default=None,
                        help="Enable focal loss with this gamma (binary or multiclass)")
    parser.add_argument("--focal_alpha", type=float, default=None,
                        help="Alpha for binary focal loss (default 0.25 if not set)")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for multiclass CE in training")
    # EMA
    parser.add_argument("--use_ema", action="store_true", help="Enable EMA of model parameters")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay in (0,1)")
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

    assert (
        args.patch_size % 8 == 0
    ), f"Patch size must be divisible by 8, got {args.patch_size}"

    # Set up task configuration
    task_cfg = get_task_config(args.taskid)
    task_type = task_cfg["task_type"]
    task_name = task_cfg["task_name"]
    num_classes = task_cfg["num_classes"]
    modalities = len(task_cfg["modalities"])
    labels = task_cfg["labels"]

    run_type = "from_scratch" if args.pretrained_weights_path is None else "finetune"
    # Defer weight transfer decision until after fusion detection
    do_weight_transfer = False
    experiment_name = f"{run_type}_{args.experiment}_{args.taskid}"

    print(f"Using num_workers: {args.num_workers}, num_devices: {args.num_devices}")
    print(f"Task type: {task_type}")
    print("ARGS:", args)

    # Set up directory structure
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, task_name)
    # If fusion dataset requested/exists, use it consistently for split discovery and loading
    fusion_dir = os.path.join(data_dir, f"{task_name}_fusion")
    # Also support passing task-specific path (e.g., .../fomo-task2). In that case, fusion lives one level up.
    if not os.path.isdir(fusion_dir):
        parent_dir = os.path.dirname(data_dir)
        candidate = os.path.join(parent_dir, f"{task_name}_fusion")
        if os.path.isdir(candidate):
            fusion_dir = candidate
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

    # Handle versioning for experiment tracking
    continue_from_most_recent = not args.new_version
    version = detect_version(save_dir, continue_from_most_recent)
    version_dir = os.path.join(save_dir, f"version_{version}")
    ensure_dir_exists(version_dir)

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

        import random
        rnd = random.Random(2025)

        def make_kfolds(items: list, k: int) -> list[list]:
            return [items[i::k] for i in range(k)] if k > 0 else [items]

        k = max(1, int(args.k_folds))
        f = max(0, int(args.fold_index))

        if task_type == "classification":
            def read_label(subj_dir: str) -> int:
                path = os.path.join(subj_dir, "label.txt")
                return int(float(open(path, "r").read().strip()))
            labels_cls = [read_label(sd) for sd in subject_dirs]
            pos = [sd for sd, y in zip(subject_dirs, labels_cls) if y == 1]
            neg = [sd for sd, y in zip(subject_dirs, labels_cls) if y == 0]
            rnd.shuffle(pos)
            rnd.shuffle(neg)
            if k > 1:
                pos_folds = make_kfolds(pos, k)
                neg_folds = make_kfolds(neg, k)
                f = min(f, k - 1)
                val_list = pos_folds[f] + neg_folds[f]
                train_list = [x for i in range(k) if i != f for x in (pos_folds[i] + neg_folds[i])]
            else:
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
        else:
            all_items = list(subject_dirs)
            rnd.shuffle(all_items)
            if k > 1:
                folds = make_kfolds(all_items, k)
                f = min(f, k - 1)
                val_list = folds[f]
                train_list = [x for i in range(k) if i != f for x in folds[i]]
            else:
                if args.split_method == "simple_train_val_split":
                    frac = float(split_param)
                    n_train = max(1, int(len(all_items) * (1 - frac)))
                    train_list = all_items[:n_train]
                    val_list = all_items[n_train:]
                else:
                    n_train = max(1, int(len(all_items) * 0.8))
                    train_list = all_items[:n_train]
                    val_list = all_items[n_train:]

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
        "learning_rate": args.learning_rate,
        "patch_size": (args.patch_size,) * 3,
        "precision": args.precision,
        "augmentation_preset": args.augmentation_preset,
        "epochs": args.epochs,
        "train_batches_per_epoch": args.train_batches_per_epoch,
        "effective_batch_size": effective_batch_size,
        
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
        
        # Trainer specific params
        "fast_dev_run": args.fast_dev_run,
        # Subject-level export
        "export_subject_probs": args.export_subject_probs,
        # Enhanced features
        "enhanced_aggregation_enabled": args.enable_enhanced_aggregation,
        "training_visualization_enabled": args.enable_training_visualization,
        # Validation-time TTA
        "val_tta": args.val_tta,
        # Scheduling & LR policy
        "scheduler": args.scheduler,
        "one_cycle_max_lr": args.one_cycle_max_lr if args.one_cycle_max_lr is not None else args.learning_rate,
        "apply_layerwise_lr_decay": args.apply_layerwise_lr_decay,
        "layerwise_lr_decay_gamma": args.layerwise_lr_decay_gamma,
        # Loss options
        "class_weights": [float(x) for x in args.class_weights.split(",")] if args.class_weights else None,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "label_smoothing": args.label_smoothing,
    }

    # Choose monitor metric
    monitor_metric = "val/loss"
    monitor_mode = "min"
    if task_type == "classification" and num_classes == 2:
        monitor_metric = "val/auroc_subject"
        monitor_mode = "max"
    elif task_type == "regression":
        # Prefer Absolute Error (MAE) for brain age regression
        monitor_metric = "val/mae"
        monitor_mode = "min"
    elif task_type == "segmentation":
        # Track dice for foreground (class 1) if available
        monitor_metric = "val/dice_1"
        monitor_mode = "max"

    # Checkpoint and early stopping callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(version_dir, "checkpoints"),
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
    )
    
    # Enhanced early stopping
    if args.enhanced_early_stopping == "robust":
        try:
            from utils.robust_early_stopping import RobustEarlyStopping
            early_stop = RobustEarlyStopping(
                primary_metric=monitor_metric,
                mode=monitor_mode,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
            )
            print("Using RobustEarlyStopping")
        except ImportError:
            print("RobustEarlyStopping not available, falling back to standard")
            early_stop = EarlyStopping(
                monitor=monitor_metric,
                mode=monitor_mode,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
            )
    elif args.enhanced_early_stopping == "adaptive":
        try:
            from utils.robust_early_stopping import AdaptiveEarlyStopping
            early_stop = AdaptiveEarlyStopping(
                primary_metric=monitor_metric,
                mode=monitor_mode,
                initial_patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
            )
            print("Using AdaptiveEarlyStopping")
        except ImportError:
            print("AdaptiveEarlyStopping not available, falling back to standard")
            early_stop = EarlyStopping(
                monitor=monitor_metric,
                mode=monitor_mode,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
            )
    else:
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
        )
        
    callbacks = [checkpoint_callback, early_stop]
    
    # Add training visualization if enabled
    if args.enable_training_visualization:
        try:
            from utils.training_visualizer import TrainingVisualizer, LiveTrainingMonitor
            visualizer = TrainingVisualizer(
                save_dir=version_dir,
                enable_live_plots=True,
                update_frequency=args.visualization_update_freq
            )
            viz_monitor = LiveTrainingMonitor(visualizer)
            callbacks.append(viz_monitor)
            print("Training visualization enabled")
        except ImportError:
            print("Training visualization not available")
    
    if args.use_swa and StochasticWeightAveraging is not None:
        callbacks.append(StochasticWeightAveraging(swa_lrs=float(args.swa_lrs)))
    if args.use_ema:
        callbacks.append(
            ModelEMACallback(
                decay=float(args.ema_decay),
                save_ema_best=True,
                monitor=monitor_metric,
                mode=monitor_mode,
                filename="ema-best",
            )
        )

    # Create logger for metrics
    yucca_logger = YuccaLogger(
        save_dir=save_dir,
        version=version,
        steps_per_epoch=args.train_batches_per_epoch,
    )
    loggers = [yucca_logger]


    # Configure augmentations based on preset
    aug_params = get_finetune_augmentation_params(args.augmentation_preset)
    augmenter = YuccaAugmentationComposer(
        patch_size=config["patch_size"],
        task_type_preset=task_type,
        parameter_dict=aug_params,
        deep_supervision=False,
    )

    # Create the data module that handles loading and batching
    # Choose dataset class by task and fusion
    if use_fusion:
        dataset_cls = FusionCLSDataset
    else:
        dataset_cls = CLSDataset

    data_module = YuccaDataModule(
        train_dataset_class=dataset_cls,  # Supports both stacked and per-modality fusion
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
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
    # Print dataset information
    print("Train dataset: ", data_module.splits_config.train(config["split_idx"]))
    print("Val dataset: ", data_module.splits_config.val(config["split_idx"]))
    print("Run type: ", run_type)
    print(
        f"Starting training with {max_iterations} max iterations over {args.epochs} epochs "
        f"with train dataset of size {train_dataset_size} datapoints and val dataset of size {val_dataset_size} "
        f"and effective batch size of {effective_batch_size}"
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
        config["multi_encoder_modalities"] = finetune_modalities
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
            print("Warning: --use_multi_encoder ignored in stacked mode; using single encoder.")

    model = BaseSupervisedModel.create(
        task_type=task_type,
        config=config,
        learning_rate=args.learning_rate,
        do_compile=args.compile,
        compile_mode="default" if args.compile_mode is None else args.compile_mode,
    )

    # Enable enhanced aggregation if requested
    if args.enable_enhanced_aggregation and hasattr(model, '_enhanced_aggregation_enabled'):
        model._enhanced_aggregation_enabled = True
        print("Enhanced subject-level aggregation enabled")

    # Optional channels-last memory format
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Create Lightning trainer
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=loggers,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        num_nodes=1,
        devices=args.num_devices,
        default_root_dir=save_dir,
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches_per_epoch,
        accumulate_grad_batches=max(1, int(args.accumulate_grad_batches)),
        gradient_clip_val=float(args.grad_clip_val) if args.grad_clip_val and args.grad_clip_val > 0 else 0.0,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
    )

    # Load pretrained weights if requested
    if do_weight_transfer:
        print("Transferring pretrained weights where available")
        print(f"Checkpoint path: {ckpt_path}")
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
            print(f"[CKPT] Loading single-encoder checkpoint: {single_ckpt}")
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

            # Debug: print loading plan
            print("[CKPT] Modality->Group mapping:")
            for m, g in mapping.items():
                print(f"  - {m} -> {g}")
            print("[CKPT] Group checkpoints:")
            for g, p in ckpt_map.items():
                print(f"  - {g}: {p}")

            merged_state = {}
            finetune_modalities = config.get("multi_encoder_modalities", [])
            for i, mod in enumerate(finetune_modalities):
                group = mapping.get(mod, None)
                if group is None:
                    print(f"Warning: No global group mapping for modality {mod}. Skipping weight load.")
                    continue
                src_ckpt = ckpt_map.get(group, None)
                if src_ckpt is None:
                    print(f"[CKPT] Missing checkpoint for group '{group}' -> modality '{mod}' will use random init.")
                    continue
                print(f"[CKPT] Loading encoder for modality '{mod}' (group='{group}') from: {src_ckpt}")
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
            print(f"[CKPT] Successfully transferred per-encoder weights for {num_successful_weights_transferred} keys (see logs for details)")
        if num_successful_weights_transferred == 0:
            print("Warning: No weights were successfully transferred; proceeding with random init.")
    else:
        print("Training from scratch, no weights will be transferred")

    # Start training
    trainer.fit(model=model, datamodule=data_module, ckpt_path="last")
    # wandb.finish()


if __name__ == "__main__":
    main()
