from typing import Optional, Dict, List
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
            "starting_filters": int(self.config.get("starting_filters", 64)),
            # Applies to most CNN-based architectures
            "conv_op": conv_op,
            # Applies to most CNN-based architectures (exceptions: UXNet)
            "norm_op": norm_op,
            # MedNeXt
            "checkpoint_style": None,
            # ensure not pretraining
            "mode": self.task_type,  # Pass task_type directly
            # Head regularization
            "cls_head_dropout_p": float(self.config.get("cls_head_dropout_p", 0.0)),
        }
        model_kwargs = filter_kwargs(model_class, model_kwargs)
        # Multi-encoder integration (optional via config)
        use_multi_encoder = bool(self.config.get("use_multi_encoder", False))
        if use_multi_encoder:
            multi_modalities = list(self.config.get("multi_encoder_modalities", []))
            modality_to_global_group = dict(self.config.get("modality_to_global_group", {}))
            global_vocab = list(self.config.get("global_vocab", ["t1","t2","flair","dwi","other"]))
            enabled_modalities = self.config.get("enabled_modalities", None)  # None means all enabled
            fusion_type = str(self.config.get("fusion_type", "masked_mean"))
            
            model_kwargs.update(
                {
                    "use_multi_encoder": True,
                    "multi_encoder_modalities": multi_modalities,
                    "multi_encoder_num_modalities_global": len(multi_modalities) if len(multi_modalities) > 0 else None,
                    "modality_to_global_group": modality_to_global_group,
                    "global_vocab": global_vocab,
                    "enabled_modalities": enabled_modalities,
                    "fusion_type": fusion_type,
                }
            )
        self.model = model_class(**model_kwargs)

        # Ensure classifier head dropout aligns with config even if kwargs were filtered
        if self.task_type in ("classification", "regression"):
            try:
                p = float(self.config.get("cls_head_dropout_p", 0.0))
                if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "dropout"):
                    self.model.decoder.dropout = (
                        torch.nn.Dropout(p=p) if p and p > 0 else torch.nn.Identity()
                    )
            except Exception:
                pass

        # Materialize head early and run a dummy forward pass to ensure parameters are initialized
        try:
            if self.task_type in ("classification", "regression"):
                # Force full model materialization with a dummy forward pass
                patch_size = tuple(self.config.get("patch_size", (32, 32, 32)))
                # Determine input shape based on encoder mode
                if bool(self.config.get("use_multi_encoder", False)):
                    # Multi-encoder expects [B, M, D, H, W]
                    num_modalities = int(
                        len(self.config.get("multi_encoder_modalities", []))
                        or self.config.get("num_modalities", 1)
                    )
                    dummy_input = torch.zeros(1, num_modalities, *patch_size, dtype=torch.float32)
                else:
                    # Single-encoder expects [B, C, D, H, W]
                    num_channels = int(self.config.get("num_modalities", 1))
                    dummy_input = torch.zeros(1, num_channels, *patch_size, dtype=torch.float32)
                
                # Set model to eval mode for materialization, then back to train
                was_training = self.model.training
                self.model.eval()
                
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                if was_training:
                    self.model.train()
                    
                logging.info("Model parameters materialized via dummy forward pass")
        except Exception as e:
            logging.warning(f"Model materialization failed (may be OK): {e}")
            pass

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Set up task-specific loss functions
        self.loss_fn_train, self.loss_fn_val = self._configure_losses()

        # Two-phase finetune support: optionally freeze encoders
        freeze_epochs = int(self.config.get("freeze_encoder_epochs", 0))
        phase1_head_lr = float(self.config.get("phase1_head_lr", self.learning_rate))
        phase2_head_lr = float(self.config.get("phase2_head_lr", self.learning_rate))
        phase2_encoder_lr = float(self.config.get("phase2_encoder_lr", self.learning_rate))

        # Check for potential AMP + freeze incompatibility
        precision = str(self.config.get("precision", "")).strip()
        if freeze_epochs > 0 and precision == "16-mixed":
            logging.warning(
                "⚠️  COMPATIBILITY WARNING: freeze_encoder_epochs > 0 with 16-mixed precision "
                "may cause AMP assertion failures. Consider using bf16-mixed, 32-true, "
                "or setting freeze_encoder_epochs=0 if training crashes."
            )

        # Store freeze configuration for use in on_train_epoch_start
        self._freeze_epochs = freeze_epochs
        self._phase1_head_lr = phase1_head_lr
        self._phase2_encoder_lr = phase2_encoder_lr
        self._phase2_head_lr = phase2_head_lr
        
        # Parameter groups: identify encoder vs head parameters
        encoder_params = []
        head_params = []
        encoder_param_names = []
        
        for name, p in self.model.named_parameters():
            if self._is_encoder_param(name):
                encoder_params.append(p)
                encoder_param_names.append(name)
            else:
                head_params.append(p)

        # Store encoder parameter names and objects for unfreezing
        self._encoder_param_names = encoder_param_names
        self._encoder_params = encoder_params
        self._head_params = head_params

        # CRITICAL: Never change requires_grad after this point
        # Instead, we'll use zero learning rates and gradient masking
        
        # Create parameter groups - ALL parameters stay requires_grad=True for AMP compatibility
        param_groups = []
        
        # Encoder parameters: use lr=0 during freeze phase, normal LR after
        if len(encoder_params) > 0:
            encoder_lr = 0.0 if freeze_epochs > 0 else phase2_encoder_lr
            param_groups.append({
                "params": encoder_params, 
                "lr": encoder_lr,
                "name": "encoder"
            })
            
        # Head parameters
        if len(head_params) > 0:
            initial_head_lr = phase1_head_lr if freeze_epochs > 0 else phase2_head_lr
            param_groups.append({
                "params": head_params, 
                "lr": initial_head_lr,
                "name": "head"
            })
            
        # Fallback: if no parameter groups created, use all parameters
        if len(param_groups) == 0:
            all_params = list(self.model.parameters())
            if len(all_params) == 0:
                raise RuntimeError("No parameters found in model")
            initial_lr = phase1_head_lr if freeze_epochs > 0 else phase2_head_lr
            param_groups = [{"params": all_params, "lr": initial_lr, "name": "all"}]

        self.optim = AdamW(
            param_groups,
            lr=self.learning_rate,  # Default LR, overridden by param groups
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            eps=self.eps,
            betas=self.betas,
        )

        # Log the freezing strategy being used with accurate parameter counts
        if freeze_epochs > 0:
            frozen_params = len(encoder_params)
            trainable_params = len(head_params) 
            total_params = frozen_params + trainable_params
            
            logging.info(f"🔒 Freeze strategy: encoder LR=0 for {freeze_epochs} epochs, then LR={phase2_encoder_lr}")
            logging.info(f"📌 Head LR: phase1={phase1_head_lr}, phase2={phase2_head_lr}")
            logging.info(f"🧠 All parameters remain requires_grad=True for AMP compatibility")
            logging.info(f"")
            logging.info(f"📊 EFFECTIVE Parameter Status During Freeze Phase:")
            logging.info(f"   {frozen_params:,} encoder params (LR=0, effectively frozen)")
            logging.info(f"   {trainable_params:,} head params (LR={phase1_head_lr}, actively training)")
            logging.info(f"   {total_params:,} total params")
            logging.info(f"")
            logging.info(f"   Note: PyTorch Lightning reports all {total_params:,} as 'trainable'")
            logging.info(f"   because requires_grad=True (needed for mixed precision),")
            logging.info(f"   but {frozen_params:,} encoder params have LR=0 so won't update.")
        else:
            logging.info(f"🔓 No encoder freezing - all parameters trainable from start")

        # Scheduler selection
        sched_choice = str(self.config.get("lr_scheduler", "cosine"))
        if sched_choice == "plateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                mode="min",
                factor=float(self.config.get("plateau_factor", 0.5)),
                patience=int(self.config.get("plateau_patience", 4)),
                threshold=float(self.config.get("plateau_threshold", 1e-3)),
                cooldown=int(self.config.get("plateau_cooldown", 0)),
                min_lr=float(self.config.get("plateau_min_lr", 1e-7)),
                verbose=False,
            )
        else:
            # Cosine with early cut-off factor of 1.15
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=int(self.trainer.max_epochs * 1.15), eta_min=1e-9
            )

        # Return the optimizer and scheduler - the loss is not returned
        if sched_choice == "plateau":
            return {
                "optimizer": self.optim,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": self.optim, "lr_scheduler": self.lr_scheduler}

    def _is_encoder_param(self, param_name: str) -> bool:
        """
        Determine if a parameter belongs to the encoder.
        Enhanced for multi-encoder and fusion architectures.
        """
        encoder_indicators = [
            ".encoders.",         # Multi-encoder: model.encoder.encoders.DWI.*
            ".fusions.",          # Fusion layers: model.encoder.fusions.*
            "encoder.",           # Standard: encoder.*
            ".encoder.",          # Standard: model.encoder.*
            "backbone.",          # Backbone: backbone.*
            ".backbone.",         # Backbone: model.backbone.*
            "feature_extractor.", # Feature extractor: feature_extractor.*
            ".feature_extractor.",# Feature extractor: model.feature_extractor.*
        ]
        
        # Check for encoder indicators
        for indicator in encoder_indicators:
            if indicator in param_name:
                return True
        
        # Check if it starts with encoder-related names
        if param_name.startswith(("encoder", "backbone", "feature_extractor")):
            return True
            
        return False

    def on_train_epoch_start(self):
        """
        Handle parameter unfreezing at the specified epoch.
        Uses LR-based freezing for full AMP compatibility.
        """
        if hasattr(self, "_freeze_epochs") and self._freeze_epochs > 0:
            if self.current_epoch == self._freeze_epochs:
                logging.info(f"🔓 UNFREEZING encoders at epoch {self.current_epoch}")
                logging.info(f"📈 Encoder LR: 0 -> {self._phase2_encoder_lr}")
                
                # Safely log head LR transition if both values are available
                if hasattr(self, "_phase1_head_lr") and hasattr(self, "_phase2_head_lr"):
                    logging.info(f"📈 Head LR: {self._phase1_head_lr} -> {self._phase2_head_lr}")
                else:
                    logging.info(f"📈 Head LR updated to: {getattr(self, '_phase2_head_lr', 'unknown')}")
                
                # Simply update learning rates - no requires_grad changes needed
                self._update_optimizer_learning_rates()
                
                # Log the new effective parameter status
                self.log_effective_parameter_status()
                
                # Reset the freeze epochs to prevent this from running again
                self._freeze_epochs = -1
            elif self.current_epoch < self._freeze_epochs:
                if self.current_epoch % 5 == 0 or self.current_epoch == 0:  # Log at start and every 5 epochs
                    logging.info(f"🔒 Encoders frozen (epoch {self.current_epoch}/{self._freeze_epochs})")
                    if self.current_epoch == 0:
                        self.log_effective_parameter_status()
    
    def _update_optimizer_learning_rates(self):
        """
        Update learning rates in the existing optimizer without recreating it.
        This is Lightning and AMP-compatible.
        """
        if not hasattr(self, "optim") or self.optim is None:
            logging.warning("⚠️  No optimizer found to update learning rates")
            return
            
        logging.info("🔄 Updating optimizer learning rates after unfreezing...")
        
        # Get target learning rates with safety checks
        target_encoder_lr = getattr(self, '_phase2_encoder_lr', 1e-5)
        target_head_lr = getattr(self, '_phase2_head_lr', 1e-4)
        
        # Update learning rates for each parameter group
        updated_groups = 0
        for group in self.optim.param_groups:
            group_name = group.get("name", "unknown")
            
            if group_name == "encoder":
                # Set encoder learning rate for unfrozen parameters
                old_lr = group["lr"]
                group["lr"] = target_encoder_lr
                logging.info(f"  ✅ Encoder group LR: {old_lr} -> {group['lr']}")
                updated_groups += 1
                
            elif group_name == "head":
                # Update head learning rate for phase 2
                old_lr = group["lr"]
                group["lr"] = target_head_lr
                logging.info(f"  ✅ Head group LR: {old_lr} -> {group['lr']}")
                updated_groups += 1
                
            else:
                # Handle unnamed groups - use head LR as default
                old_lr = group["lr"]
                group["lr"] = target_head_lr
                logging.info(f"  ✅ {group_name} group LR: {old_lr} -> {group['lr']} (defaulted to head LR)")
                updated_groups += 1
        
        logging.info(f"✅ Successfully updated {updated_groups} optimizer parameter groups")
    
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
        
        # Fix tensor shape mismatch for regression tasks
        if output.dim() > 1 and output.size(-1) == 1:
            output = output.squeeze(-1)
            
        loss = self.loss_fn_train(output, target)

        if self.deep_supervision and hasattr(output, "__iter__"):
            # If deep_supervision is enabled, output and target will be a list of (downsampled) tensors.
            # We only need the original ground truth and its corresponding prediction which is always the first entry in each list.
            output_for_metrics = output[0]
            target_for_metrics = target[0]
        else:
            output_for_metrics = output
            target_for_metrics = target

        metrics = self.compute_metrics(self.train_metrics, output_for_metrics, target_for_metrics)
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
        
        # Fix tensor shape mismatch for regression tasks
        if output.dim() > 1 and output.size(-1) == 1:
            output = output.squeeze(-1)
            
        loss = self.loss_fn_val(output, target)
        
        # Handle deep supervision for metrics
        if self.deep_supervision and hasattr(output, "__iter__"):
            output_for_metrics = output[0]
            target_for_metrics = target[0]
        else:
            output_for_metrics = output
            target_for_metrics = target
            
        metrics = self.compute_metrics(self.val_metrics, output_for_metrics, target_for_metrics)
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
        """Load state dict with handling for different model architectures"""
        # First we filter out layers that have changed in size
        # This is often the case in the output layer.
        # If we are finetuning on a task with a different number of classes
        # than the pretraining task, the # output channels will have changed.
        old_params = copy.deepcopy(self.state_dict())
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if (k in old_params) and (old_params[k].shape == state_dict[k].shape)
        }
        rejected_keys_new = [k for k in state_dict.keys() if k not in old_params]
        rejected_keys_shape = [
            k for k in state_dict.keys() if old_params[k].shape != state_dict[k].shape
        ]
        rejected_keys_data = []

        # Here there's also potential to implement custom loading functions.
        # E.g. to load 2D pretrained models into 3D by repeating or something like that.

        # Now keep track of the # of layers with succesful weight transfers
        successful = 0
        unsuccessful = 0
        super().load_state_dict(state_dict, *args, **kwargs)
        new_params = self.state_dict()
        for param_name, p1, p2 in zip(
            old_params.keys(), old_params.values(), new_params.values()
        ):
            # If more than one param in layer is NE (not equal) to the original weights we've successfully loaded new weights.
            if p1.data.ne(p2.data).sum() > 0:
                successful += 1
            else:
                unsuccessful += 1
                if (
                    param_name not in rejected_keys_new
                    and param_name not in rejected_keys_shape
                ):
                    rejected_keys_data.append(param_name)

        logging.warning(
            f"Succesfully transferred weights for {successful}/{successful+unsuccessful} layers"
        )
        logging.warning(
            f"Rejected the following keys:\n"
            f"Not in old dict: {rejected_keys_new}.\n"
            f"Wrong shape: {rejected_keys_shape}.\n"
            f"Post check not succesful: {rejected_keys_data}."
        )

        return successful

    def set_enabled_modalities(self, enabled_modalities: List[str]) -> None:
        """
        Dynamically enable/disable modalities for ablation studies.
        Only works for multi-encoder models.
        
        Args:
            enabled_modalities: List of modality names to enable
        """
        if not hasattr(self.model, 'encoder') or not hasattr(self.model.encoder, 'set_enabled_modalities'):
            raise ValueError("Model does not support modality switching (not a multi-encoder model)")
        
        self.model.encoder.set_enabled_modalities(enabled_modalities)
        
    def get_modality_status(self) -> Dict[str, bool]:
        """
        Get current enable/disable status of all modalities.
        Only works for multi-encoder models.
        
        Returns:
            Dictionary mapping modality names to their enabled status
        """
        if not hasattr(self.model, 'encoder') or not hasattr(self.model.encoder, 'get_modality_status'):
            raise ValueError("Model does not support modality switching (not a multi-encoder model)")
        
        return self.model.encoder.get_modality_status()
    
    def enable_modality(self, modality_name: str) -> None:
        """Enable a specific modality."""
        if not hasattr(self.model, 'encoder') or not hasattr(self.model.encoder, 'enable_modality'):
            raise ValueError("Model does not support modality switching (not a multi-encoder model)")
        
        self.model.encoder.enable_modality(modality_name)
    
    def disable_modality(self, modality_name: str) -> None:
        """Disable a specific modality."""
        if not hasattr(self.model, 'encoder') or not hasattr(self.model.encoder, 'disable_modality'):
            raise ValueError("Model does not support modality switching (not a multi-encoder model)")
        
        self.model.encoder.disable_modality(modality_name)
    
    def get_effective_parameter_counts(self) -> Dict[str, int]:
        """
        Get accurate parameter counts considering current freeze state.
        
        Returns:
            Dictionary with 'frozen', 'trainable', and 'total' parameter counts
        """
        if not hasattr(self, '_freeze_epochs') or self._freeze_epochs <= 0:
            # No freezing active
            total = sum(p.numel() for p in self.parameters())
            return {
                'frozen': 0,
                'trainable': total,
                'total': total,
                'freeze_active': False
            }
        
        # During freeze phase
        if self.current_epoch < self._freeze_epochs:
            encoder_count = sum(p.numel() for p in getattr(self, '_encoder_params', []))
            head_count = sum(p.numel() for p in getattr(self, '_head_params', []))
            return {
                'frozen': encoder_count,
                'trainable': head_count, 
                'total': encoder_count + head_count,
                'freeze_active': True
            }
        else:
            # After unfreezing
            total = sum(p.numel() for p in self.parameters())
            return {
                'frozen': 0,
                'trainable': total,
                'total': total,
                'freeze_active': False
            }
    
    def log_effective_parameter_status(self) -> None:
        """Log current effective parameter status with clear explanation."""
        counts = self.get_effective_parameter_counts()
        
        if counts['freeze_active']:
            logging.info(f"📊 Current Effective Parameter Status:")
            logging.info(f"   🔒 {counts['frozen']:,} params effectively FROZEN (encoder, LR=0)")
            logging.info(f"   🔓 {counts['trainable']:,} params actively TRAINING (head, LR>0)")
            logging.info(f"   📝 Total: {counts['total']:,} params")
            logging.info(f"   ⚡ Note: All params have requires_grad=True for AMP, but frozen params won't update")
        else:
            logging.info(f"📊 Current Parameter Status:")
            logging.info(f"   🔓 {counts['trainable']:,} params actively TRAINING")
            logging.info(f"   📝 Total: {counts['total']:,} params")

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
