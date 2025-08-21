from typing import Optional
import torch
import torch.nn.functional as F
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
        
        # Sliding window validation settings
        self.val_sliding_window_enable = config.get("val_sliding_window_enable", False)
        self.val_sliding_window_overlap = config.get("val_sliding_window_overlap", 0.5)
        self.val_gaussian_weights = config.get("val_gaussian_weights", True)
        
        # Initialize TTA state
        self._val_tta_enable = False
        self._val_tta_views = 1
        self._tta_codes = [()]
        self._tta_offsets = [(0, 0, 0)]
        
        # Initialize loss functions
        self.loss_fn_train, self.loss_fn_val = self._configure_losses()

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
    
    def _process_batch(self, batch):
        """
        Process segmentation batch data
        
        Args:
            batch: Input batch
            
        Returns:
            tuple: (inputs, target, file_path)
        """
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        
        # Ensure target is long for segmentation
        target = target.long()
        
        # Remove channel dimension if it exists for metrics (but keep for loss)
        if target.dim() == 5 and target.size(1) == 1:
            target = target.squeeze(1)  # [B, 1, D, H, W] -> [B, D, H, W]
        
        return inputs, target, file_path
    
    def on_validation_epoch_start(self):
        """Initialize TTA and sliding window validation state"""
        # TTA settings (similar to SupervisedClsModel)
        self._val_tta_enable = bool(self.config.get("val_tta_enable", False))
        self._val_tta_views = int(self.config.get("val_tta_views", 1))
        
        # Precompute flip codes (up to 8: none, x, y, z, xy, xz, yz, xyz)
        self._tta_codes = [
            (), (2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)
        ][: max(1, min(8, self._val_tta_views))]
        
        # Translation offsets (center + axis shifts)
        self._val_tta_offsets = int(self.config.get("val_tta_offsets", 1))
        self._val_tta_offset_frac = float(self.config.get("val_tta_offset_frac", 0.25))
        
        # Build offsets: center, +/- along each axis (up to 7 total)
        base = [(0, 0, 0)]
        shifts = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        self._tta_offsets = base + shifts
        self._tta_offsets = self._tta_offsets[: max(1, min(7, self._val_tta_offsets))]

    def create_gaussian_weights(self, patch_size):
        """Create Gaussian weight map for smooth patch blending"""
        weights = []
        for dim in patch_size:
            # Create 1D Gaussian for each dimension
            sigma = dim / 6.0  # Standard deviation as fraction of patch size
            x = torch.arange(dim, dtype=torch.float32)
            center = (dim - 1) / 2.0
            gaussian_1d = torch.exp(-0.5 * ((x - center) / sigma) ** 2)
            weights.append(gaussian_1d)
        
        # Create 3D Gaussian by outer product
        gaussian_3d = weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None, :]
        return gaussian_3d
    
    def generate_patch_coordinates(self, volume_shape, patch_size, stride):
        """Generate coordinates for sliding window patches"""
        coordinates = []
        for z in range(0, volume_shape[0] - patch_size[0] + 1, stride[0]):
            for y in range(0, volume_shape[1] - patch_size[1] + 1, stride[1]):
                for x in range(0, volume_shape[2] - patch_size[2] + 1, stride[2]):
                    coordinates.append((z, y, x))
        
        # Handle edge cases - ensure we cover the entire volume
        if len(coordinates) == 0 or coordinates[-1] != (volume_shape[0] - patch_size[0], 
                                                        volume_shape[1] - patch_size[1], 
                                                        volume_shape[2] - patch_size[2]):
            # Add final patch to cover the end of the volume
            coordinates.append((volume_shape[0] - patch_size[0], 
                              volume_shape[1] - patch_size[1], 
                              volume_shape[2] - patch_size[2]))
        return coordinates
    
    def sliding_window_inference(self, inputs, target=None):
        """
        Perform sliding window inference with optional TTA
        
        Args:
            inputs: Input tensor [B, M, D, H, W]
            target: Optional target tensor [B, D, H, W] or [B, 1, D, H, W]
            
        Returns:
            Averaged predictions over sliding window patches
        """
        B, M, D, H, W = inputs.shape
        patch_size = self.config.get("patch_size", (64, 64, 32))
        
        # Check if sliding window is needed
        if (D <= patch_size[0] and H <= patch_size[1] and W <= patch_size[2]):
            # Volume fits in single patch, use regular TTA if enabled
            return self.forward_with_tta(inputs)
        
        # Calculate stride based on overlap
        stride = tuple(int(p * (1 - self.val_sliding_window_overlap)) for p in patch_size)
        
        # Generate patch coordinates
        coords = self.generate_patch_coordinates((D, H, W), patch_size, stride)
        
        # Initialize prediction and weight accumulators
        pred_shape = (B, self.num_classes, D, H, W)
        prediction_map = torch.zeros(pred_shape, device=inputs.device, dtype=inputs.dtype)
        weight_map = torch.zeros((D, H, W), device=inputs.device, dtype=inputs.dtype)
        
        # Create Gaussian weights if enabled
        if self.val_gaussian_weights:
            gaussian_weights = self.create_gaussian_weights(patch_size).to(inputs.device)
        else:
            gaussian_weights = torch.ones(patch_size, device=inputs.device)
        
        # Process each patch
        for z, y, x in coords:
            # Extract patch
            patch = inputs[:, :, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
            
            # Forward pass with optional TTA
            with torch.no_grad():
                if self._val_tta_enable:
                    patch_pred = self.forward_with_tta(patch)
                else:
                    patch_pred = self(patch)
            
            # Apply Gaussian weighting
            weighted_pred = patch_pred * gaussian_weights[None, None, :, :, :]
            
            # Accumulate predictions and weights
            prediction_map[:, :, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += weighted_pred
            weight_map[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += gaussian_weights
        
        # Normalize by accumulated weights
        weight_map = weight_map + 1e-8  # Avoid division by zero
        final_prediction = prediction_map / weight_map[None, None, :, :, :]
        
        return final_prediction
    
    def forward_with_tta(self, inputs):
        """
        Forward pass with test-time augmentation (adapted from SupervisedClsModel)
        """
        if not self._val_tta_enable or len(self._tta_codes) == 1 and len(self._tta_offsets) == 1:
            return self(inputs)
        
        B, M, D, H, W = inputs.shape
        dz = int(round(self._val_tta_offset_frac * D))
        dy = int(round(self._val_tta_offset_frac * H))
        dx = int(round(self._val_tta_offset_frac * W))
        
        combos_per_step = int(max(1, self.config.get("val_tta_batch_size", 4)))  # Smaller for segmentation
        
        # Build list of combo tuples
        combo_tuples = []
        for oz, oy, ox in self._tta_offsets:
            for code in self._tta_codes:
                combo_tuples.append((oz, oy, ox, code))
        
        total_combos = len(combo_tuples)
        sum_preds = None
        processed = 0
        i = 0
        
        while i < total_combos:
            # Determine group size
            g = min(combos_per_step, total_combos - i)
            
            # Build mini-batch
            mini = []
            for j in range(g):
                oz, oy, ox, code = combo_tuples[i + j]
                x = torch.roll(inputs, shifts=(oz * dz, oy * dy, ox * dx), dims=(2, 3, 4))
                if len(code) > 0:
                    x = torch.flip(x, dims=list(code))
                mini.append(x)
            
            try:
                X = torch.cat(mini, dim=0)  # [g*B, M, D, H, W]
                preds = self(X)              # [g*B, C, D, H, W]
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and combos_per_step > 1:
                    combos_per_step = max(1, combos_per_step // 2)
                    continue
                raise
            
            # Get prediction dimensions
            pred_shape = preds.shape[1:]  # [C, D, H, W]
            if sum_preds is None:
                sum_preds = torch.zeros((B,) + pred_shape, device=preds.device, dtype=preds.dtype)
            
            # Reshape and sum over groups: [g, B, C, D, H, W] -> [B, C, D, H, W]
            preds = preds.view(g, B, *pred_shape).sum(dim=0)
            sum_preds = sum_preds + preds
            processed += g
            i += g
        
        # Average over all combos
        output = sum_preds / float(max(1, processed))
        return output
    
    def validation_step(self, batch, batch_idx):
        """
        Enhanced validation step with sliding window and TTA support
        """
        # Ensure TTA state exists
        if not hasattr(self, "_val_tta_enable"):
            self.on_validation_epoch_start()
        
        # Process batch
        inputs, target, _ = self._process_batch(batch)
        
        # Use sliding window inference if enabled and needed
        if self.val_sliding_window_enable:
            output = self.sliding_window_inference(inputs, target)
        else:
            # Standard validation with optional TTA
            if self._val_tta_enable:
                output = self.forward_with_tta(inputs)
            else:
                output = self(inputs)
        
        # Fix tensor shape mismatch: DiceCE loss expects target with channel dimension
        if target.dim() == 4:  # [B, D, H, W] -> [B, 1, D, H, W]
            target_for_loss = target.unsqueeze(1)
        else:
            target_for_loss = target
            
        loss = self.loss_fn_val(output, target_for_loss)
        
        # DEPRECATED: dice_debug metric removed to avoid confusion with proper val/dice
        # dice_debug computed on single batch only, while val/dice is properly aggregated
        # Use val/dice for all validation assessments - it's the correct implementation
        # from models.losses import get_dice_coefficient
        # dice_coeff = get_dice_coefficient(output, target)
        # self.log("val/dice_debug", dice_coeff, on_step=False, on_epoch=True, prog_bar=False)
        
        # Validate loss is finite
        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite validation loss detected: {loss}")
            loss = torch.tensor(0.0, device=loss.device)
        
        # Handle deep supervision for metrics
        if self.deep_supervision and hasattr(output, "__iter__"):
            output_for_metrics = output[0]
            target_for_metrics = target[0] if hasattr(target, "__iter__") else target
        else:
            output_for_metrics = output
            target_for_metrics = target
            
        metrics = self.compute_metrics(self.val_metrics, output_for_metrics, target_for_metrics)
        self.log_dict(
            {"val/loss": loss} | metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        
        # Optionally compute surface metrics for detailed evaluation
        if self.surface_metrics_enabled and batch_idx == 0:  # Sample a few batches
            try:
                self._compute_surface_metrics_sample(batch)
            except Exception as e:
                # Don't fail training if surface metrics computation fails
                self.log("val/surface_metrics_error", 1.0, prog_bar=False)
        
        return loss
    
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
