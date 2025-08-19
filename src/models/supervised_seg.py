from typing import Optional
import torch
import numpy as np
from torchmetrics import MetricCollection
from torchmetrics.classification import Dice, F1Score

from yucca.modules.optimization.loss_functions.deep_supervision import (
    DeepSupervisionLoss,
)
from models.losses import CorrectedDiceCE, get_dice_coefficient

from models.supervised_base import BaseSupervisedModel


class SupervisedSegModel(BaseSupervisedModel):
    """
    Supervised model for segmentation tasks.
    Inherits from BaseSupervisedModel and implements segmentation-specific functionality.
    Supports Dice, F1, and surface-based metrics including Normal Surface Distance (NSD).
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
        super().__init__(
            config=config,
            learning_rate=learning_rate,
            do_compile=do_compile,
            compile_mode=compile_mode,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            eps=eps,
            betas=betas,
            deep_supervision=deep_supervision,
        )
        
        # Track surface metrics computation during validation
        self.surface_metrics_enabled = config.get("surface_metrics_enabled", True)

    def forward(self, inputs):
        """Segmentation forward pass with support for batch dicts.

        Accepts either a tensor [B,M,D,H,W] or a batch dict containing keys
        'image' and optional 'modality_mask'. This mirrors test expectations
        that call model.forward(batch).
        """
        # Unpack batch dict if provided
        if isinstance(inputs, dict):
            x = inputs.get("image")
            mask = inputs.get("modality_mask", None)
            # UNet supports optional mask argument; pass when available
            return self.model(x, mask=mask)
        # Tensor path
        return self.model(inputs)

    def _configure_metrics(self, prefix: str):
        """
        Configure segmentation-specific metrics including Dice and F1.
        Surface metrics like NSD are computed separately during validation.

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of segmentation metrics
        """
        return MetricCollection(
            {
                f"{prefix}/dice": Dice(
                    num_classes=self.num_classes,
                    ignore_index=0,       # Focus on foreground class performance
                    average='macro',      # Average across non-ignored classes (just foreground)
                ),
                f"{prefix}/f1": F1Score(
                    task='multiclass',
                    num_classes=self.num_classes,
                    ignore_index=0 if self.num_classes > 1 else None,
                    average='macro',
                ),
            },
        )

    def _configure_losses(self):
        """
        Configure segmentation-specific loss functions with corrected DiceCE

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """
        # Use corrected DiceCE loss with stronger emphasis on Dice component
        # for better foreground class learning
        loss_fn_train = CorrectedDiceCE(
            soft_dice_kwargs={"apply_softmax": True},
            weight_ce=1,      # Cross-entropy weight
            weight_dice=2     # Emphasize dice loss for better segmentation
        )
        loss_fn_val = CorrectedDiceCE(
            soft_dice_kwargs={"apply_softmax": True},
            weight_ce=1,
            weight_dice=2
        )

        if self.deep_supervision:
            # Note: DeepSupervisionLoss may need verification with corrected loss
            from yucca.modules.optimization.loss_functions.nnUNet_losses import DeepSupervisionLoss
            loss_fn_train = DeepSupervisionLoss(loss_fn_train, weights=None)

        return loss_fn_train, loss_fn_val

    def compute_metrics(self, metrics, output, target, ignore_index: int | None = None):
        """Compute segmentation metrics (Dice coefficient and F1).

        - Ensures integer targets
        - Uses torchmetrics F1Score which handles both logits and predictions
        - Keeps aggregated metrics keys (e.g., 'val/dice', 'val/f1') present
        - Supports ignore_index in signature for test compatibility
        """
        # Ensure target is in correct shape for metrics [B, D, H, W]
        if target.dim() == 5 and target.size(1) == 1:
            target = target.squeeze(1)  # Remove channel dimension for metrics
            
        # Targets must be integer
        if target.dtype != torch.long:
            target = target.long()

        # Default ignore_index if not provided and task is multiclass
        if ignore_index is None and self.num_classes > 1:
            ignore_index = 0

        # Determine if we have logits or already converted predictions
        has_logits = (isinstance(output, torch.Tensor) and output.dim() >= 2 and 
                     output.shape[1] == getattr(self, "num_classes", output.shape[1]) and
                     output.dtype.is_floating_point)
        
        # Convert logits to predictions for Dice (torchmetrics Dice expects class predictions)
        if has_logits:
            pred_classes = torch.argmax(output, dim=1)
        else:
            pred_classes = output

        # Compute metrics individually to handle different input requirements
        result_metrics = {}
        
        for metric_name, metric in metrics.items():
            if 'dice' in metric_name.lower():
                # Dice expects class predictions [B, D, H, W]
                metric_result = metric(pred_classes, target)
            elif 'f1' in metric_name.lower():
                # torchmetrics F1Score handles both logits and predictions well
                if has_logits:
                    metric_result = metric(output, target)  # Use logits
                else:
                    metric_result = metric(pred_classes, target)  # Use predictions
            else:
                # Default: use predictions
                metric_result = metric(pred_classes, target)
            
            # Store result (torchmetrics F1Score returns scalar with average='macro')
            result_metrics[metric_name] = metric_result
        
        return result_metrics
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step with optional surface metrics computation.
        """
        # Call parent validation step for standard metrics
        result = super().validation_step(batch, batch_idx)
        
        # Optionally compute surface metrics for detailed evaluation
        if self.surface_metrics_enabled and batch_idx == 0:  # Sample a few batches
            try:
                self._compute_surface_metrics_sample(batch)
            except Exception as e:
                # Don't fail training if surface metrics computation fails
                self.log("val/surface_metrics_error", 1.0, prog_bar=False)
        
        return result
    
    def _compute_surface_metrics_sample(self, batch):
        """
        Compute surface metrics on a sample batch for monitoring.
        This is for logging purposes during training.
        """
        try:
            from yucca.functional.evaluation.surface_metrics import get_surface_metrics_for_label
            import nibabel as nib
            
            # Get predictions and targets
            with torch.no_grad():
                output = self.forward(batch)
                if isinstance(output, list):  # Handle deep supervision
                    output = output[0]
                
                pred = torch.softmax(output, dim=1)
                pred_classes = torch.argmax(pred, dim=1)
                
                # Convert to numpy for surface metrics
                pred_np = pred_classes.cpu().numpy()
                target_np = batch["label"].cpu().numpy()
                
                # Compute surface metrics for foreground class (class 1)
                if pred_np.max() > 0 and target_np.max() > 0:
                    # Create dummy nibabel images for surface metrics
                    pred_nii = nib.Nifti1Image(pred_np[0].astype(np.uint8), affine=np.eye(4))
                    target_nii = nib.Nifti1Image(target_np[0].astype(np.uint8), affine=np.eye(4))
                    
                    surface_metrics = get_surface_metrics_for_label(
                        target_nii, pred_nii, label=1, as_binary=True
                    )
                    
                    # Log surface metrics
                    for metric_name, value in surface_metrics.items():
                        self.log(f"val/surface_{metric_name.lower().replace(' ', '_')}", 
                                float(value), prog_bar=False)
                        
        except Exception as e:
            # Silently skip surface metrics if computation fails
            pass
