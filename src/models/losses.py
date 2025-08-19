"""
Custom loss functions for FOMO segmentation tasks.

This module provides corrected loss functions to address sign issues
in the yucca library's DiceCE implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from yucca.modules.optimization.loss_functions.nnUNet_losses import get_tp_fp_fn_tn
from yucca.modules.optimization.loss_functions.CE import CE


class CorrectedSoftDiceLoss(nn.Module):
    """
    Corrected Soft Dice Loss that returns (1 - dice) instead of (-dice).
    
    This fixes the sign issue in yucca's SoftDiceLoss where negative dice
    coefficients were returned, causing training instability.
    """
    
    def __init__(self, apply_softmax=True, batch_dice=False, do_bg=True, smooth=1e-5):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_softmax is True:
            x = F.softmax(x, 1)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        # FIXED: Return (1 - dice) instead of (-dice) for proper loss behavior
        return 1.0 - dc


class CorrectedDiceCE(nn.Module):
    """
    Corrected DiceCE Loss with proper sign convention.
    
    This combines:
    - Corrected Dice Loss: (1 - dice_coefficient) 
    - Cross-Entropy Loss: standard CE loss
    
    Final loss = weight_dice * (1 - dice) + weight_ce * ce_loss
    """
    
    def __init__(self, 
                 soft_dice_kwargs=None,
                 ce_kwargs=None,
                 weight_ce=1,
                 weight_dice=1,
                 log_dice=False,
                 ignore_label=None):
        super().__init__()
        
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {}
        if ce_kwargs is None:
            ce_kwargs = {}
            
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        
        # Use corrected dice loss
        self.dc = CorrectedSoftDiceLoss(**soft_dice_kwargs)
        
        # Standard cross-entropy loss
        if ignore_label is not None:
            ce_kwargs["reduction"] = "none"
        self.ce = CE()

    def forward(self, net_output, target):
        """
        Forward pass for corrected DiceCE loss.
        
        Args:
            net_output: Model predictions [B, num_classes, D, H, W]
            target: Ground truth targets [B, 1, D, H, W] 
            
        Returns:
            Combined loss value (positive, decreasing with better predictions)
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, "not implemented for one hot encoding"
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        # Compute dice loss: (1 - dice_coefficient)
        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        
        # Optional log transform (usually not needed)
        if self.log_dice:
            # Ensure dc_loss > 0 before taking log
            dc_loss = -torch.log(torch.clamp(1 - dc_loss, min=1e-8))

        # Compute cross-entropy loss
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        
        # Apply ignore label masking if needed
        if self.ignore_label is not None and self.weight_ce != 0:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / (mask.sum() + 1e-8)

        # Combine losses with proper signs
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        
        return result


def get_dice_coefficient(predictions, targets, smooth=1e-5):
    """
    Compute dice coefficient for monitoring (not for loss).
    
    Args:
        predictions: Model logits [B, num_classes, D, H, W]
        targets: Ground truth [B, D, H, W] or [B, 1, D, H, W]
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient (0.0 to 1.0)
    """
    # Handle target dimensions
    if targets.dim() == 5 and targets.size(1) == 1:
        targets = targets.squeeze(1)  # [B, D, H, W]
    
    # Convert logits to predictions
    if predictions.dim() == 5:  # [B, num_classes, D, H, W]
        pred_classes = torch.argmax(predictions, dim=1)  # [B, D, H, W]
    else:
        pred_classes = predictions
    
    # Compute dice for foreground class (class 1)
    pred_fg = (pred_classes == 1).float()
    target_fg = (targets == 1).float()
    
    intersection = (pred_fg * target_fg).sum()
    union = pred_fg.sum() + target_fg.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()
