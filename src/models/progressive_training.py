"""
Progressive training strategies for fusion models.
Implements gradual unfreezing and adaptive learning rate schedules.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import math


class ProgressiveUnfreezingScheduler:
    """
    Progressive unfreezing scheduler for multi-encoder fusion training.
    
    Instead of freezing all encoders then unfreezing at once, gradually
    introduces encoder parameters to training for better stability.
    """
    
    def __init__(
        self,
        total_epochs: int = 500,
        fusion_warmup_epochs: int = 3,
        encoder_warmup_epochs: int = 5,
        base_head_lr: float = 1e-3,
        base_fusion_lr: float = 1e-4,
        base_encoder_lr: float = 5e-5,
        min_lr_factor: float = 0.1
    ):
        self.total_epochs = total_epochs
        self.fusion_warmup_epochs = fusion_warmup_epochs
        self.encoder_warmup_epochs = encoder_warmup_epochs
        self.base_head_lr = base_head_lr
        self.base_fusion_lr = base_fusion_lr
        self.base_encoder_lr = base_encoder_lr
        self.min_lr_factor = min_lr_factor
        
        # Define training phases
        self.phases = {
            "head_only": (0, fusion_warmup_epochs),
            "head_fusion": (fusion_warmup_epochs, encoder_warmup_epochs),
            "full_training": (encoder_warmup_epochs, total_epochs)
        }
    
    def get_learning_rates(self, epoch: int) -> Dict[str, float]:
        """
        Get learning rates for each component at given epoch.
        
        Args:
            epoch: Current training epoch (0-indexed)
            
        Returns:
            Dict with keys: "encoder_lr", "fusion_lr", "head_lr"
        """
        if epoch < self.phases["head_only"][1]:
            # Phase 1: Head only training
            return {
                "encoder_lr": 0.0,
                "fusion_lr": 0.0,
                "head_lr": self.base_head_lr
            }
        elif epoch < self.phases["head_fusion"][1]:
            # Phase 2: Head + Fusion training
            # Gradual warmup of fusion LR
            warmup_progress = (epoch - self.phases["head_fusion"][0]) / (
                self.phases["head_fusion"][1] - self.phases["head_fusion"][0]
            )
            fusion_lr = self.base_fusion_lr * warmup_progress
            
            return {
                "encoder_lr": 0.0,
                "fusion_lr": fusion_lr,
                "head_lr": self.base_head_lr * 0.8  # Slightly reduce head LR
            }
        else:
            # Phase 3: Full training with cosine annealing
            progress = (epoch - self.phases["full_training"][0]) / (
                self.phases["full_training"][1] - self.phases["full_training"][0]
            )
            
            # Cosine annealing for stable training
            cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
            lr_factor = self.min_lr_factor + (1 - self.min_lr_factor) * cos_factor
            
            return {
                "encoder_lr": self.base_encoder_lr * lr_factor,
                "fusion_lr": self.base_fusion_lr * lr_factor,
                "head_lr": self.base_head_lr * 0.5 * lr_factor  # Lower head LR in full training
            }
    
    def apply_to_optimizer(self, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        """
        Apply learning rates to optimizer parameter groups.
        
        Assumes optimizer has parameter groups in order: [encoder, fusion, head]
        """
        lrs = self.get_learning_rates(epoch)
        
        # Map to parameter groups (assumes specific order)
        param_group_lrs = [lrs["encoder_lr"], lrs["fusion_lr"], lrs["head_lr"]]
        
        for param_group, lr in zip(optimizer.param_groups, param_group_lrs):
            param_group["lr"] = lr
    
    def get_phase_info(self, epoch: int) -> Dict[str, Any]:
        """Get current phase information for logging."""
        lrs = self.get_learning_rates(epoch)
        
        if epoch < self.phases["head_only"][1]:
            phase = "head_only"
        elif epoch < self.phases["head_fusion"][1]:
            phase = "head_fusion" 
        else:
            phase = "full_training"
            
        return {
            "phase": phase,
            "epoch": epoch,
            **lrs,
            "active_params": self._get_active_param_info(lrs)
        }
    
    def _get_active_param_info(self, lrs: Dict[str, float]) -> Dict[str, bool]:
        """Determine which parameter groups are active (LR > 0)."""
        return {
            "encoder_active": lrs["encoder_lr"] > 0,
            "fusion_active": lrs["fusion_lr"] > 0,
            "head_active": lrs["head_lr"] > 0
        }


class AdaptiveFusionLoss:
    """
    Adaptive loss weighting for fusion training stability.
    
    Combines main task loss with regularization terms that adapt
    based on training progress and model confidence.
    """
    
    def __init__(
        self,
        uncertainty_weight: float = 0.01,
        consistency_weight: float = 0.001,
        sparsity_weight: float = 0.0001,
        warmup_epochs: int = 10
    ):
        self.uncertainty_weight = uncertainty_weight
        self.consistency_weight = consistency_weight
        self.sparsity_weight = sparsity_weight
        self.warmup_epochs = warmup_epochs
        
        self.current_epoch = 0
    
    def update_epoch(self, epoch: int) -> None:
        """Update current epoch for weight scheduling."""
        self.current_epoch = epoch
    
    def compute_loss(
        self,
        main_loss: torch.Tensor,
        model: nn.Module,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with adaptive regularization.
        
        Args:
            main_loss: Primary task loss (e.g., CrossEntropy)
            model: The fusion model for extracting regularization terms
            predictions: Model predictions (for uncertainty estimation)
            targets: Ground truth targets
            
        Returns:
            Dict containing individual loss components and total loss
        """
        losses = {"main_loss": main_loss}
        total_loss = main_loss
        
        # Weight scheduling based on training progress
        warmup_factor = min(1.0, self.current_epoch / self.warmup_epochs)
        
        # Uncertainty regularization (encourage confident predictions)
        if predictions is not None and self.uncertainty_weight > 0:
            uncertainty_loss = self._compute_uncertainty_loss(predictions)
            weighted_uncertainty = self.uncertainty_weight * warmup_factor * uncertainty_loss
            losses["uncertainty_loss"] = uncertainty_loss
            total_loss = total_loss + weighted_uncertainty
        
        # Fusion consistency regularization
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'fusions'):
            consistency_loss = self._compute_fusion_consistency_loss(model.encoder.fusions)
            weighted_consistency = self.consistency_weight * warmup_factor * consistency_loss
            losses["consistency_loss"] = consistency_loss
            total_loss = total_loss + weighted_consistency
        
        # Sparsity regularization on fusion weights
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'fusions'):
            sparsity_loss = self._compute_sparsity_loss(model.encoder.fusions)
            weighted_sparsity = self.sparsity_weight * warmup_factor * sparsity_loss
            losses["sparsity_loss"] = sparsity_loss
            total_loss = total_loss + weighted_sparsity
        
        losses["total_loss"] = total_loss
        return losses
    
    def _compute_uncertainty_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Encourage confident predictions by penalizing high entropy.
        
        Args:
            predictions: Model logits [B, num_classes] or probabilities
            
        Returns:
            Uncertainty loss scalar
        """
        if predictions.dim() == 1:
            # Binary case - convert to probabilities
            probs = torch.sigmoid(predictions)
            probs = torch.stack([1 - probs, probs], dim=-1)
        else:
            # Multi-class case
            probs = torch.softmax(predictions, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy.mean()
    
    def _compute_fusion_consistency_loss(self, fusion_modules: nn.ModuleList) -> torch.Tensor:
        """
        Encourage consistent fusion weights across scales.
        
        Args:
            fusion_modules: List of fusion modules from different scales
            
        Returns:
            Consistency loss scalar
        """
        consistency_loss = 0.0
        num_pairs = 0
        
        # Extract fusion weights from modules that have them
        fusion_weights = []
        for fusion in fusion_modules:
            if hasattr(fusion, 'modality_weights'):
                fusion_weights.append(fusion.modality_weights)
            elif hasattr(fusion, 'gamma'):
                # Use gamma as proxy for modality importance
                fusion_weights.append(fusion.gamma.mean(dim=-1))  # Average across channels
        
        # Compute pairwise consistency
        for i in range(len(fusion_weights)):
            for j in range(i + 1, len(fusion_weights)):
                # Normalize weights to make them comparable
                w1 = torch.softmax(fusion_weights[i], dim=-1)
                w2 = torch.softmax(fusion_weights[j], dim=-1)
                
                # KL divergence between weight distributions
                kl_div = torch.sum(w1 * torch.log((w1 + 1e-8) / (w2 + 1e-8)))
                consistency_loss = consistency_loss + kl_div
                num_pairs += 1
        
        return consistency_loss / max(num_pairs, 1)
    
    def _compute_sparsity_loss(self, fusion_modules: nn.ModuleList) -> torch.Tensor:
        """
        Encourage sparsity in fusion parameters to prevent overfitting.
        
        Args:
            fusion_modules: List of fusion modules
            
        Returns:
            Sparsity loss scalar
        """
        sparsity_loss = 0.0
        num_params = 0
        
        for fusion in fusion_modules:
            # L1 regularization on learnable parameters
            for name, param in fusion.named_parameters():
                if 'weight' in name and param.requires_grad:
                    sparsity_loss = sparsity_loss + torch.sum(torch.abs(param))
                    num_params += param.numel()
        
        return sparsity_loss / max(num_params, 1)


class FusionModelWrapper:
    """
    Wrapper for fusion models that provides easy progressive training integration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        total_epochs: int = 500,
        progressive_config: Optional[Dict] = None,
        adaptive_loss_config: Optional[Dict] = None
    ):
        self.model = model
        self.total_epochs = total_epochs
        
        # Initialize progressive scheduler
        prog_config = progressive_config or {}
        self.scheduler = ProgressiveUnfreezingScheduler(
            total_epochs=total_epochs,
            **prog_config
        )
        
        # Initialize adaptive loss
        loss_config = adaptive_loss_config or {}
        self.adaptive_loss = AdaptiveFusionLoss(**loss_config)
        
        self.current_epoch = 0
    
    def step_epoch(self, epoch: int, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """
        Step to next epoch with progressive training updates.
        
        Args:
            epoch: Current epoch
            optimizer: Model optimizer
            
        Returns:
            Phase information for logging
        """
        self.current_epoch = epoch
        self.adaptive_loss.update_epoch(epoch)
        
        # Update learning rates
        self.scheduler.apply_to_optimizer(optimizer, epoch)
        
        # Get phase info for logging
        phase_info = self.scheduler.get_phase_info(epoch)
        
        return phase_info
    
    def compute_loss(
        self,
        main_loss: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive loss with regularization.
        
        Args:
            main_loss: Primary task loss
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dict of loss components
        """
        return self.adaptive_loss.compute_loss(
            main_loss=main_loss,
            model=self.model,
            predictions=predictions,
            targets=targets
        )
