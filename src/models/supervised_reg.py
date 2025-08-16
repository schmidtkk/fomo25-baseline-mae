from typing import Optional
import torch
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
        # Store loss type from config for brain age regression optimization
        self.loss_type = config.get("loss_type", "mse")  # mse, mae, huber
        self.age_normalization = config.get("age_normalization", True)
        self.age_mean = config.get("age_mean", 50.0)  # Approximate brain age mean
        self.age_std = config.get("age_std", 15.0)   # Approximate brain age std
        
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
            print("🧠 Using MAE loss for robust brain age regression")
        elif self.loss_type == "huber":
            # Huber loss - combines MSE and MAE benefits
            loss_fn = torch.nn.HuberLoss(delta=1.0)  # delta=1 year for age prediction
            print("🧠 Using Huber loss for robust brain age regression")
        else:  # Default MSE
            loss_fn = torch.nn.MSELoss()
            print("🧠 Using MSE loss for brain age regression")
            
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
