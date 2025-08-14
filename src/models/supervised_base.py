from typing import Optional
import torch
from torch.optim import AdamW
import copy
import logging

import lightning as L
from yucca.pipeline.preprocessing import YuccaPreprocessor
from yucca.functional.utils.kwargs import filter_kwargs
from batchgenerators.utilities.file_and_folder_operations import join
from models import networks


class BaseSupervisedModel(L.LightningModule):
    """
    Base class for supervised models (segmentation, classification, regression).
    Implements common functionality and defines abstract methods that subclasses must implement.
    """

    def __init__(
        self,
        config: dict = {},
        learning_rate: float = 1e-3,
        do_compile: Optional[bool] = False,
        compile_mode: Optional[str] = "default",
        weight_decay: float = 3e-5,
        amsgrad: bool = False,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
        deep_supervision: bool = False,
    ):
        super().__init__()

        # Keep full config for later reference
        self.config = config

        self.num_classes = config["num_classes"]
        self.num_modalities = config["num_modalities"]
        self.patch_size = config["patch_size"]
        self.plans = config.get("plans", {})
        self.model_name = config["model_name"]
        self.version_dir = config["version_dir"]
        self.task_type = config["task_type"]  # Added task_type property

        self.sliding_window_prediction = True
        self.sliding_window_overlap = 0.5  # nnUNet default
        self.test_time_augmentation = False
        self.progress_bar = True

        self.do_compile = do_compile
        self.compile_mode = compile_mode

        # Loss
        self.deep_supervision = deep_supervision

        # Optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.eps = eps
        self.betas = betas

        # Set up metrics in subclasses
        self.train_metrics = self._configure_metrics(prefix="train")
        self.val_metrics = self._configure_metrics(prefix="val")

        self.save_hyperparameters()
        self.load_model()

        self.model = (
            torch.compile(self.model, mode=self.compile_mode)
            if self.do_compile
            else self.model
        )

    def _configure_metrics(self, prefix: str):
        """
        Configure metrics specific to the task type.
        Must be implemented by subclasses.

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of metrics for the task
        """
        raise NotImplementedError("Subclasses must implement _configure_metrics")

    def _configure_losses(self):
        """
        Configure loss functions specific to the task type.
        Must be implemented by subclasses.

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """
        raise NotImplementedError("Subclasses must implement _configure_losses")

    def load_model(self):
        """Load the appropriate model architecture"""
        print(f"Loading Model: 3D {self.model_name}")
        model_class = getattr(networks, self.model_name)

        print("Found model class: ", model_class)

        conv_op = torch.nn.Conv3d
        norm_op = torch.nn.InstanceNorm3d
        print("MODALITIES", self.num_modalities)

        # Pass task_type directly to UNet without mapping
        model_kwargs = {
            # Applies to all models
            "input_channels": self.num_modalities,
            "num_classes": self.num_classes,
            "output_channels": self.num_classes,
            "deep_supervision": self.deep_supervision,
            # Applies to most CNN-based architectures
            "conv_op": conv_op,
            # Applies to most CNN-based architectures (exceptions: UXNet)
            "norm_op": norm_op,
            # MedNeXt
            "checkpoint_style": None,
            # ensure not pretraining
            "mode": self.task_type,  # Pass task_type directly
        }
        model_kwargs = filter_kwargs(model_class, model_kwargs)
        # Multi-encoder integration (optional via config)
        use_multi_encoder = bool(self.config.get("use_multi_encoder", False))
        if use_multi_encoder:
            multi_modalities = list(self.config.get("multi_encoder_modalities", []))
            modality_to_global_group = dict(self.config.get("modality_to_global_group", {}))
            global_vocab = list(self.config.get("global_vocab", ["t1","t2","flair","dwi","other"]))
            model_kwargs.update(
                {
                    "use_multi_encoder": True,
                    "multi_encoder_modalities": multi_modalities,
                    "multi_encoder_num_modalities_global": len(multi_modalities) if len(multi_modalities) > 0 else None,
                    "modality_to_global_group": modality_to_global_group,
                    "global_vocab": global_vocab,
                }
            )
        self.model = model_class(**model_kwargs)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Set up task-specific loss functions
        self.loss_fn_train, self.loss_fn_val = self._configure_losses()

        # Two-phase finetune support: optionally freeze encoders
        freeze_epochs = int(self.config.get("freeze_encoder_epochs", 0))

        # Parameter groups: identify encoder vs head (optional layer-wise decay)
        apply_layerwise = bool(self.config.get("apply_layerwise_lr_decay", False))
        layerwise_gamma = float(self.config.get("layerwise_lr_decay_gamma", 0.0) or 0.0)
        encoder_named_params = []
        head_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if ".encoders." in name or name.startswith("encoder"):
                encoder_named_params.append((name, p))
            else:
                head_params.append(p)
        # Default: single encoder group; if layer-wise enabled, create multiple groups with decayed LR
        param_groups = []
        if apply_layerwise and len(encoder_named_params) > 0 and layerwise_gamma > 0:
            # Split into 4 depth groups by name order as a simple heuristic
            encoder_named_params.sort(key=lambda x: x[0])
            num = len(encoder_named_params)
            bins = [encoder_named_params[i * num // 4:(i + 1) * num // 4] for i in range(4)]
            for depth, params_at_depth in enumerate(bins):
                if not params_at_depth:
                    continue
                lr_mult = (layerwise_gamma ** depth)
                param_groups.append({"params": [p for _, p in params_at_depth], "lr_mult": lr_mult})
        else:
            param_groups.append({"params": [p for _, p in encoder_named_params], "lr_mult": 1.0})

        # Initial LR: freeze encoders if requested
        encoder_params_flat = [p for _, p in encoder_named_params]
        if freeze_epochs > 0 and len(encoder_params_flat) > 0:
            for p in encoder_params_flat:
                p.requires_grad = False
            param_groups = [
                {"params": head_params},
            ]
        else:
            if len(param_groups) == 1:
                # Single encoder group
                param_groups = [
                    {"params": param_groups[0]["params"]},
                    {"params": head_params},
                ]
            else:
                # Layer-wise groups with lr multipliers
                lr_groups = []
                for g in param_groups:
                    lr_groups.append({"params": g["params"]})
                param_groups = lr_groups + [{"params": head_params}]

        self.optim = AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            eps=self.eps,
            betas=self.betas,
        )

        # Scheduler selection
        scheduler_name = str(self.config.get("scheduler", "cosine"))
        if scheduler_name == "cosine_restarts":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optim, T_0=max(1, int(self.trainer.max_epochs / 3)), T_mult=2, eta_min=1e-9
            )
        elif scheduler_name == "one_cycle":
            steps_per_epoch = int(self.config.get("train_batches_per_epoch", 100))
            max_lr = float(self.config.get("one_cycle_max_lr", self.learning_rate))
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                max_lr=max_lr,
                epochs=int(self.trainer.max_epochs),
                steps_per_epoch=max(1, steps_per_epoch),
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1e4,
            )
        elif scheduler_name == "none":
            self.lr_scheduler = None
        else:
            # CosineAnnealingLR with early cut-off factor
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=int(self.trainer.max_epochs * 1.15), eta_min=1e-9
            )

        # Store freeze schedule in state
        self._freeze_epochs = freeze_epochs
        self._warmup_epochs = 5

        # Return the optimizer and scheduler - the loss is not returned
        if self.lr_scheduler is None:
            return {"optimizer": self.optim}
        # Lightning can accept a dict or list; pass scheduler with interval configured for OneCycle
        sched = self.lr_scheduler
        if isinstance(sched, torch.optim.lr_scheduler.OneCycleLR):
            return {"optimizer": self.optim, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return {"optimizer": self.optim, "lr_scheduler": sched}

    def on_train_epoch_start(self):
        # Unfreeze encoders after freeze window
        if hasattr(self, "_freeze_epochs") and self._freeze_epochs > 0:
            if self.current_epoch == self._freeze_epochs:
                for name, p in self.model.named_parameters():
                    if ".encoders." in name or name.startswith("encoder"):
                        p.requires_grad = True
                # Rebuild param groups with desired LRs
                encoder_params = []
                head_params = []
                for name, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if ".encoders." in name or name.startswith("encoder"):
                        encoder_params.append(p)
                    else:
                        head_params.append(p)
                self.optim.param_groups.clear()
                if len(encoder_params) > 0:
                    self.optim.add_param_group({"params": encoder_params})
                if len(head_params) > 0:
                    self.optim.add_param_group({"params": head_params})

    def forward(self, inputs):
        """Forward pass through the model"""
        return self.model(inputs)

    def _process_batch(self, batch):
        """Process batch data - can be overridden by subclasses if needed"""
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        return inputs, target, file_path

    def training_step(self, batch, _batch_idx):
        """Training step"""
        inputs, target, _ = self._process_batch(batch)

        output = self(inputs)
        loss = self.loss_fn_train(output, target)

        if self.deep_supervision and hasattr(output, "__iter__"):
            # If deep_supervision is enabled, output and target will be a list of (downsampled) tensors.
            # We only need the original ground truth and its corresponding prediction which is always the first entry in each list.
            output = output[0]
            target = target[0]

        metrics = self.compute_metrics(self.train_metrics, output, target)
        self.log_dict(
            {"train/loss": loss} | metrics,
            prog_bar=self.progress_bar,
            logger=True,
        )

        return loss

    def validation_step(self, batch, _batch_idx):
        """Validation step"""
        inputs, target, _ = self._process_batch(batch)

        output = self(inputs)
        loss = self.loss_fn_val(output, target)
        metrics = self.compute_metrics(self.val_metrics, output, target)
        self.log_dict(
            {"val/loss": loss} | metrics,
            prog_bar=self.progress_bar,
            logger=True,
        )

    def on_predict_start(self):
        """Set up for prediction"""
        self.preprocessor = YuccaPreprocessor(join(self.version_dir, "hparams.yaml"))

    def predict_step(self, batch, _batch_idx, _dataloader_idx=0):
        """Prediction step"""
        case, case_id = batch
        (
            case_preprocessed,
            case_properties,
        ) = self.preprocessor.preprocess_case_for_inference(
            case, self.patch_size, self.sliding_window_prediction
        )

        predictions = self.model.predict(
            data=case_preprocessed,
            mode="3D",
            mirror=self.test_time_augmentation,
            overlap=self.sliding_window_overlap,
            patch_size=self.patch_size,
            sliding_window_prediction=self.sliding_window_prediction,
            device=self.device,
        )
        predictions, case_properties = self.preprocessor.reverse_preprocessing(
            predictions, case_properties
        )
        return {
            "predictions": predictions,
            "properties": case_properties,
            "case_id": case_id[0],
        }

    def compute_metrics(self, metrics, output, target, ignore_index=None):
        """
        Compute task-specific metrics.
        Should be implemented/extended by subclasses for task-specific metrics.
        """
        raise NotImplementedError("Subclasses must implement compute_metrics")

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict with robust reporting focused on actually attempted keys.

        - Compute rejections against the provided state dict before filtering
        - Load only keys present and shape-compatible
        - Report success/unchanged among the attempted keys (not all model params)
        """
        from torch.nn.parameter import UninitializedParameter

        def _safe_same_shape(a, b) -> bool:
            try:
                if isinstance(a, UninitializedParameter) or isinstance(b, UninitializedParameter):
                    return False
                return hasattr(a, "shape") and hasattr(b, "shape") and a.shape == b.shape
            except Exception:
                return False

        old_params = copy.deepcopy(self.state_dict())
        provided = copy.deepcopy(state_dict)

        # Determine rejects relative to current model params (avoid touching .shape on uninitialized)
        rejected_keys_new = [k for k in provided.keys() if k not in old_params]
        rejected_keys_shape = [k for k in provided.keys() if (k in old_params) and (not _safe_same_shape(old_params[k], provided[k]))]

        # Filter to attempted keys: present and shape-compatible
        filtered = {k: v for k, v in provided.items() if (k in old_params) and _safe_same_shape(old_params[k], v)}

        # Load
        super().load_state_dict(filtered, *args, **kwargs)

        # Post-check success for attempted keys only
        new_params = self.state_dict()
        successful_keys = []
        unchanged_after_load = []
        for k in filtered.keys():
            before = old_params[k]
            after = new_params[k]
            try:
                changed = before.data.ne(after.data).sum() > 0
            except Exception:
                changed = False
            if changed:
                successful_keys.append(k)
            else:
                unchanged_after_load.append(k)

        num_attempted = len(filtered)
        num_successful = len(successful_keys)
        num_unchanged = len(unchanged_after_load)

        # Logging
        logging.info(
            f"Successfully transferred {num_successful}/{num_attempted} compatible layers; "
            f"{len(rejected_keys_new)} keys not found; {len(rejected_keys_shape)} with mismatched shape."
        )
        if rejected_keys_new or rejected_keys_shape:
            logging.info(
                "Some keys were rejected (set log level DEBUG to see full lists)."
            )
            logging.debug(f"Not in model: {rejected_keys_new}")
            logging.debug(f"Wrong shape: {rejected_keys_shape}")
        if num_unchanged > 0:
            logging.info(
                f"{num_unchanged} loaded keys did not change values (likely identical to current weights)."
            )
            logging.debug(f"Unchanged after load: {unchanged_after_load}")

        return num_successful

    @staticmethod
    def create(task_type, config, **kwargs):
        """
        Factory method to create the appropriate model based on task type

        Args:
            task_type: Type of task (segmentation, classification, regression)
            config: Configuration dictionary
            **kwargs: Additional arguments for the model

        Returns:
            BaseSupervisedModel: Instance of appropriate model subclass
        """
        if task_type == "segmentation":
            from models.supervised_seg import SupervisedSegModel

            return SupervisedSegModel(config=config, **kwargs)
        elif task_type == "classification":
            from models.supervised_cls import SupervisedClsModel

            return SupervisedClsModel(config=config, **kwargs)
        elif task_type == "regression":
            from models.supervised_reg import SupervisedRegModel

            return SupervisedRegModel(config=config, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
