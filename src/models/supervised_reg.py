from typing import Optional, Dict, List
import torch
import os
import logging
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef

from models.supervised_base import BaseSupervisedModel


"""
Refactor: use built-in PearsonCorrCoef with epoch-level compute for robustness and DDP support.
We keep MAE in a MetricCollection and handle Pearson via dedicated metrics per phase.
"""


class SupervisedRegModel(BaseSupervisedModel):
    """
    Supervised model for regression tasks.
    Inherits from BaseSupervisedModel and implements regression-specific functionality.
    Enhanced for brain age regression with proper metrics and loss functions.
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
    ):
        # Store config for TTA and other needs (PyTorch Lightning seems to lose self.config)
        self._reg_config = dict(config) if config else {}
        
        # Store loss type from config for brain age regression optimization
        self.loss_type = self._reg_config.get("loss_type", "mse")  # mse, mae, huber
        self.age_normalization = self._reg_config.get("age_normalization", True)
        self.age_mean = self._reg_config.get("age_mean", 50.0)  # Approximate brain age mean
        self.age_std = self._reg_config.get("age_std", 15.0)   # Approximate brain age std
        
        super().__init__(
            config=config,
            learning_rate=learning_rate,
            do_compile=do_compile,
            compile_mode=compile_mode,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            eps=eps,
            betas=betas,
            deep_supervision=False,  # Regression doesn't use deep supervision
        )
        # One-time warnings for NaN corr replacement
        self._warned_nan_train_corr = False
        self._warned_nan_val_corr = False

        # Dedicated Pearson correlation metrics per phase (epoch-level)
        self.pearson_train = PearsonCorrCoef()
        self.pearson_val = PearsonCorrCoef()

    def on_validation_epoch_start(self):
        """Initialize TTA configuration for validation epoch - adapted from SupervisedClsModel"""
        # Ensure loss functions are configured if not already done
        if not hasattr(self, 'loss_fn_val'):
            self.loss_fn_train, self.loss_fn_val = self._configure_losses()
        
        # TTA configuration for regression validation using stored config
        # Use setattr to ensure PyTorch Lightning doesn't interfere
        setattr(self, '_val_tta_enable', bool(self._reg_config.get("val_tta_enable", False)))
        setattr(self, '_val_tta_views', int(self._reg_config.get("val_tta_views", 1)))
        
        # Precompute flip codes (up to 8: none, x, y, z, xy, xz, yz, xyz)
        tta_codes = [
            (), (2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)
        ][: max(1, min(8, self._val_tta_views))]
        setattr(self, '_tta_codes', tta_codes)
        
        # Deterministic translation offsets (center + axis shifts)
        val_tta_offsets = int(self._reg_config.get("val_tta_offsets", 1))
        val_tta_offset_frac = float(self._reg_config.get("val_tta_offset_frac", 0.25))
        setattr(self, '_val_tta_offsets', val_tta_offsets)
        setattr(self, '_val_tta_offset_frac', val_tta_offset_frac)
        
        # Build offsets: center, +/- along each axis (up to 7 total)
        base = [(0, 0, 0)]
        shifts = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        tta_offsets = base + shifts
        tta_offsets = tta_offsets[: max(1, min(7, val_tta_offsets))]
        setattr(self, '_tta_offsets', tta_offsets)

    def validation_step(self, batch, _batch_idx):
        """Validation step with TTA support for regression tasks"""
        # Ensure TTA state exists for unit tests that call validation_step directly
        if not hasattr(self, "_val_tta_enable"):
            self.on_validation_epoch_start()
            
        inputs, target, file_path = self._process_batch(batch)
        
        # Apply TTA if enabled, otherwise use standard forward pass
        if not self._val_tta_enable or (len(self._tta_codes) == 1 and len(self._tta_offsets) == 1):
            output = self(inputs)
        else:
            output = self._compute_tta_prediction(inputs)
        
        # Fix tensor shape mismatch for regression tasks
        if output.dim() > 1 and output.size(-1) == 1:
            output = output.squeeze(-1)
            
        loss = self.loss_fn_val(output, target)
        
        # Validate loss is finite
        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite validation loss detected: {loss}")
            loss = torch.tensor(0.0, device=loss.device)
        
        metrics = self.compute_metrics(self.val_metrics, output, target)
        self.log_dict(
            {"val/loss": loss} | metrics,
            prog_bar=self.progress_bar,
            logger=True,
        )

    def _compute_tta_prediction(self, inputs):
        """Compute TTA prediction by averaging across augmented views for regression"""
        B, M, D, H, W = inputs.shape
        dz = int(round(self._val_tta_offset_frac * D))
        dy = int(round(self._val_tta_offset_frac * H))
        dx = int(round(self._val_tta_offset_frac * W))
        
        # Use smaller batch size for regression models (they tend to be memory-intensive)
        combos_per_step = int(max(1, self._reg_config.get("val_tta_batch_size", 4)))
        
        # Build list of combo tuples to avoid holding all tensors at once
        combo_tuples = []
        for oz, oy, ox in self._tta_offsets:
            for code in self._tta_codes:
                combo_tuples.append((oz, oy, ox, code))
        
        total_combos = len(combo_tuples)
        sum_preds = None
        processed = 0
        i = 0
        
        while i < total_combos:
            # Determine group size and build a stacked mini-batch
            g = min(combos_per_step, total_combos - i)
            
            # Build mini-batch on-the-fly to limit memory usage
            mini = []
            for j in range(g):
                oz, oy, ox, code = combo_tuples[i + j]
                x = torch.roll(inputs, shifts=(oz * dz, oy * dy, ox * dx), dims=(2, 3, 4))
                if len(code) > 0:
                    x = torch.flip(x, dims=list(code))
                mini.append(x)
            
            try:
                X = torch.cat(mini, dim=0)  # [g*B, M, D, H, W]
                preds = self(X)             # [g*B, C] where C=1 for regression
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and combos_per_step > 1:
                    # Reduce group size and retry without crashing validation
                    combos_per_step = max(1, combos_per_step // 2)
                    continue
                raise
            
            # Handle regression output shape
            if preds.dim() > 1 and preds.size(-1) == 1:
                preds = preds.squeeze(-1)  # [g*B]
            
            # Initialize sum_preds with correct shape
            if sum_preds is None:
                if preds.dim() == 1:
                    sum_preds = torch.zeros(B, device=preds.device, dtype=preds.dtype)
                else:
                    sum_preds = torch.zeros(B, preds.size(-1), device=preds.device, dtype=preds.dtype)
            
            # Reshape predictions and sum over group dimension
            if preds.dim() == 1:
                preds = preds.view(g, B)  # [g, B]
                preds_sum = preds.sum(dim=0)  # [B]
            else:
                preds = preds.view(g, B, -1)  # [g, B, C]
                preds_sum = preds.sum(dim=0)  # [B, C]
            
            sum_preds = sum_preds + preds_sum
            processed += g
            i += g
        
        # Average over all combos
        avg_preds = sum_preds / float(max(1, processed))
        
        # Ensure output shape matches regular forward pass
        if avg_preds.dim() == 1:
            avg_preds = avg_preds.unsqueeze(-1)  # [B] -> [B, 1] for regression
        
        return avg_preds

    def _configure_metrics(self, prefix: str):
        """
        Configure regression-specific metrics with epoch-level Pearson correlation for brain age

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of regression metrics including correlation
        """
        # Keep MAE inside a collection; manage corr via dedicated metric objects.
        return MetricCollection({f"{prefix}/mae": MeanAbsoluteError()})

    def _configure_losses(self):
        """
        Configure regression-specific loss functions optimized for brain age

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """
        if self.loss_type == "mae":
            # MAE loss - more robust to age outliers than MSE
            loss_fn = torch.nn.L1Loss()
            logging.info("Using MAE loss for robust brain age regression")
        elif self.loss_type == "huber":
            # Huber loss - combines MSE and MAE benefits
            loss_fn = torch.nn.HuberLoss(delta=1.0)  # delta=1 year for age prediction
            logging.info("Using Huber loss for robust brain age regression")
        else:  # Default MSE
            loss_fn = torch.nn.MSELoss()
            logging.info("Using MSE loss for brain age regression")
            
        return loss_fn, loss_fn

    def _process_batch(self, batch):
        """
        Process regression batch data with optional age normalization

        Args:
            batch: Input batch

        Returns:
            tuple: (inputs, target, file_path)
        """
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        
        # Keep target as float for regression tasks
        target = target.float()
        
        # Apply age normalization if enabled (recommended for brain age)
        if self.age_normalization:
            target = (target - self.age_mean) / self.age_std
            
        return inputs, target, file_path

    def compute_metrics(self, metrics, output, target, ignore_index=None):
        """
        Compute regression metrics with denormalization for proper age evaluation

        Args:
            metrics: Metrics collection
            output: Model output
            target: Ground truth
            ignore_index: Index to ignore in metrics (not used in regression)

        Returns:
            dict: Dictionary of computed metrics
        """
        # Ensure tensor shapes match: squeeze output to match target shape
        if output.dim() > 1 and output.size(-1) == 1:
            output = output.squeeze(-1)
            
        # Denormalize predictions and targets for proper age metric calculation (for MAE only)
        if self.age_normalization:
            output_denorm = output * self.age_std + self.age_mean
            target_denorm = target * self.age_std + self.age_mean
        else:
            output_denorm = output
            target_denorm = target
        
        # Update MAE on denormalized scale for interpretability
        current = metrics(output_denorm, target_denorm)

        # Update Pearson (epoch-level) on normalized scale for stability
        try:
            if metrics is self.train_metrics:
                self.pearson_train.update(output.detach().float().flatten(), target.detach().float().flatten())
            elif metrics is self.val_metrics:
                self.pearson_val.update(output.detach().float().flatten(), target.detach().float().flatten())
        except Exception:
            pass
        return current

    def on_train_epoch_end(self):
        """Log epoch-level correlation once per epoch and reset metrics."""
        try:
            # Log MAE
            vals = self.train_metrics.compute()
            for k, v in vals.items():
                self.log(k, v, prog_bar=False, logger=True)

            # Log Pearson corr using dedicated metric
            if hasattr(self, "pearson_train"):
                try:
                    corr = self.pearson_train.compute()
                    # Replace NaN/Inf with 0.0 to avoid missing values
                    if not torch.isfinite(corr):
                        if not self._warned_nan_train_corr:
                            import logging
                            logging.warning("train/corr was NaN or Inf (likely constant predictions/targets). Logging 0.0 instead.")
                            self._warned_nan_train_corr = True
                        corr = torch.tensor(0.0, device=self.device)
                    self.log("train/corr", corr, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
                except Exception:
                    pass
        finally:
            # Reset for next epoch
            self.train_metrics.reset()
            if hasattr(self, "pearson_train"):
                self.pearson_train.reset()

    def on_validation_epoch_end(self):
        """Log epoch-level correlation for validation once per epoch and reset metrics."""
        try:
            # Log MAE
            vals = self.val_metrics.compute()
            for k, v in vals.items():
                self.log(k, v, prog_bar=False, logger=True)

            # Log Pearson corr using dedicated metric
            if hasattr(self, "pearson_val"):
                try:
                    corr = self.pearson_val.compute()
                    # Replace NaN/Inf with 0.0 to ensure the metric is visible & monitored
                    if not torch.isfinite(corr):
                        if not self._warned_nan_val_corr:
                            import logging
                            logging.warning("val/corr was NaN or Inf (likely constant predictions/targets). Logging 0.0 instead.")
                            self._warned_nan_val_corr = True
                        corr = torch.tensor(0.0, device=self.device)
                    self.log("val/corr", corr, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
                except Exception:
                    pass
        finally:
            # Reset for next epoch
            self.val_metrics.reset()
            if hasattr(self, "pearson_val"):
                self.pearson_val.reset()
