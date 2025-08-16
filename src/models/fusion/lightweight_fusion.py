"""
Lightweight fusion alternatives to the complex AttentionFusion3D.
Designed for stability, scalability, and reduced parameter count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LearnableWeightedFusion3D(nn.Module):
    """
    Simple learnable weighted fusion - most stable option.
    
    Uses learned per-modality weights with softmax normalization.
    Minimal parameters, maximum stability.
    """
    
    def __init__(
        self, 
        channels: int, 
        num_modalities: int | None = None, 
        use_gamma: bool = True, 
        use_null_token: bool = False
    ):
        super().__init__()
        self.channels = channels
        
        # Pre-alignment (reuse from MaskedMean for consistency)
        self.pre_gn = nn.GroupNorm(1, channels, affine=True)
        self.pre_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        
        # Learnable modality weights
        if num_modalities is not None:
            self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        else:
            self.modality_weights = None
            
        # Gamma scaling per modality (if requested)
        self.use_gamma = use_gamma and (num_modalities is not None)
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.ones(num_modalities, channels))
        
        # Null token for missing modalities
        self.use_null = use_null_token
        if self.use_null:
            self.null_token = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        
        # Post-fusion components
        self.post_scale = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.post_bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.eps = 1e-6

    def _pre_align(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-alignment normalization and projection."""
        x = self.pre_gn(x)
        x = self.pre_proj(x)
        rms = x.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).add(self.eps).sqrt()
        return x / rms

    def forward(
        self, 
        feats_list: List[torch.Tensor], 
        mask: torch.Tensor, 
        modality_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Args:
            feats_list: List of [B,C,D,H,W] features, one per modality
            mask: [B,M] presence mask (1=present, 0=missing)
            modality_ids: Optional list of global modality indices for gamma scaling
            
        Returns:
            fused: [B,C,D,H,W] fused features
        """
        assert len(feats_list) > 0, "Empty feats_list provided to fusion"
        B, C, D, H, W = feats_list[0].shape
        M = len(feats_list)
        
        # Pre-alignment for all modalities
        aligned = []
        for i, f in enumerate(feats_list):
            f = self._pre_align(f)
            
            # Apply gamma scaling if enabled
            if self.use_gamma and modality_ids is not None:
                gid = modality_ids[i]
                f = f * self.gamma[gid].view(1, C, 1, 1, 1)
            
            aligned.append(f)
        
        # Stack aligned features: [B, M, C, D, H, W]
        feats_stack = torch.stack(aligned, dim=1)
        
        # Apply presence mask
        m = mask.view(B, M, 1, 1, 1, 1).type_as(feats_stack)
        feats_stack = feats_stack * m
        
        # Add null tokens for missing modalities if enabled
        if self.use_null:
            feats_stack = feats_stack + (1 - m) * self.null_token
        
        # Compute modality weights (global or learned)
        if self.modality_weights is not None and modality_ids is not None:
            # Use learned weights for present modalities
            present_weights = torch.stack([self.modality_weights[gid] for gid in modality_ids])
            weights = F.softmax(present_weights, dim=0)  # [M]
        else:
            # Equal weighting
            weights = torch.ones(M, device=feats_stack.device) / M
        
        # Apply weights and mask
        weights = weights.view(1, M, 1, 1, 1, 1) * m
        
        # Weighted sum with safe normalization
        weighted_feats = feats_stack * weights
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(self.eps)  # [B, 1, 1, 1, 1, 1]
        fused = weighted_feats.sum(dim=1) / weight_sum.squeeze(1)  # [B, C, D, H, W]
        
        # Post-fusion scaling and bias
        fused = fused * self.post_scale + self.post_bias
        
        return fused


class ChannelGatedFusion3D(nn.Module):
    """
    Channel-wise gating fusion - moderate complexity, good performance.
    
    Uses global pooling + gating to learn channel importance per modality combination.
    """
    
    def __init__(
        self, 
        channels: int, 
        num_modalities: int | None = None, 
        use_gamma: bool = True, 
        use_null_token: bool = False,
        reduction_ratio: int = 4
    ):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Pre-alignment
        self.pre_gn = nn.GroupNorm(1, channels, affine=True)
        self.pre_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        
        # Channel gating mechanism
        hidden_channels = max(channels // reduction_ratio, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # [B, M*C, 1, 1, 1]
            nn.Conv3d(channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # Gamma scaling per modality (if requested)
        self.use_gamma = use_gamma and (num_modalities is not None)
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.ones(num_modalities, channels))
        
        # Null token for missing modalities
        self.use_null = use_null_token
        if self.use_null:
            self.null_token = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        
        # Post-fusion components
        self.post_scale = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.post_bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.eps = 1e-6

    def _pre_align(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-alignment normalization and projection."""
        x = self.pre_gn(x)
        x = self.pre_proj(x)
        rms = x.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).add(self.eps).sqrt()
        return x / rms

    def forward(
        self, 
        feats_list: List[torch.Tensor], 
        mask: torch.Tensor, 
        modality_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        assert len(feats_list) > 0, "Empty feats_list provided to fusion"
        B, C, D, H, W = feats_list[0].shape
        M = len(feats_list)
        
        # Pre-alignment for all modalities
        aligned = []
        for i, f in enumerate(feats_list):
            f = self._pre_align(f)
            
            # Apply gamma scaling if enabled
            if self.use_gamma and modality_ids is not None:
                gid = modality_ids[i]
                f = f * self.gamma[gid].view(1, C, 1, 1, 1)
            
            aligned.append(f)
        
        # Stack and apply mask
        feats_stack = torch.stack(aligned, dim=1)  # [B, M, C, D, H, W]
        m = mask.view(B, M, 1, 1, 1, 1).type_as(feats_stack)
        feats_stack = feats_stack * m
        
        # Add null tokens if enabled
        if self.use_null:
            feats_stack = feats_stack + (1 - m) * self.null_token
        
        # Compute mean for gating (across present modalities)
        cnt = mask.sum(dim=1, keepdim=True).clamp_min(1.0).view(B, 1, 1, 1, 1, 1)
        mean_feats = feats_stack.sum(dim=1) / cnt.squeeze(1)  # [B, C, D, H, W]
        
        # Apply channel gating
        gate_weights = self.gate(mean_feats)  # [B, C, 1, 1, 1]
        gated_mean = mean_feats * gate_weights
        
        # Post-fusion scaling and bias
        fused = gated_mean * self.post_scale + self.post_bias
        
        return fused


class UncertaintyWeightedFusion3D(nn.Module):
    """
    Uncertainty-weighted fusion - learns per-modality confidence.
    
    Each modality predicts its own uncertainty, used for dynamic weighting.
    More sophisticated but still lightweight compared to full attention.
    """
    
    def __init__(
        self, 
        channels: int, 
        num_modalities: int | None = None, 
        use_gamma: bool = True, 
        use_null_token: bool = False
    ):
        super().__init__()
        self.channels = channels
        
        # Pre-alignment
        self.pre_gn = nn.GroupNorm(1, channels, affine=True)
        self.pre_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        
        # Uncertainty estimation per modality
        self.uncertainty_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # [B, C, 1, 1, 1]
            nn.Conv3d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, 1, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Gamma scaling per modality (if requested)
        self.use_gamma = use_gamma and (num_modalities is not None)
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.ones(num_modalities, channels))
        
        # Null token for missing modalities
        self.use_null = use_null_token
        if self.use_null:
            self.null_token = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        
        # Post-fusion components
        self.post_scale = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.post_bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.eps = 1e-6

    def _pre_align(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-alignment normalization and projection."""
        x = self.pre_gn(x)
        x = self.pre_proj(x)
        rms = x.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).add(self.eps).sqrt()
        return x / rms

    def forward(
        self, 
        feats_list: List[torch.Tensor], 
        mask: torch.Tensor, 
        modality_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        assert len(feats_list) > 0, "Empty feats_list provided to fusion"
        B, C, D, H, W = feats_list[0].shape
        M = len(feats_list)
        
        # Pre-alignment and uncertainty estimation
        aligned = []
        uncertainties = []
        
        for i, f in enumerate(feats_list):
            f = self._pre_align(f)
            
            # Apply gamma scaling if enabled
            if self.use_gamma and modality_ids is not None:
                gid = modality_ids[i]
                f = f * self.gamma[gid].view(1, C, 1, 1, 1)
            
            # Estimate uncertainty for this modality
            uncertainty = self.uncertainty_head(f)  # [B, 1, 1, 1, 1]
            
            aligned.append(f)
            uncertainties.append(uncertainty)
        
        # Stack features and uncertainties
        feats_stack = torch.stack(aligned, dim=1)  # [B, M, C, D, H, W]
        uncertainty_stack = torch.stack(uncertainties, dim=1)  # [B, M, 1, 1, 1, 1]
        
        # Apply presence mask
        m = mask.view(B, M, 1, 1, 1, 1).type_as(feats_stack)
        feats_stack = feats_stack * m
        uncertainty_stack = uncertainty_stack * m.squeeze(-1).squeeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Add null tokens if enabled
        if self.use_null:
            feats_stack = feats_stack + (1 - m) * self.null_token
        
        # Compute inverse uncertainty weights (lower uncertainty = higher weight)
        inv_uncertainty = 1.0 / (uncertainty_stack + self.eps)  # [B, M, 1, 1, 1, 1]
        
        # Normalize weights across modalities
        weight_sum = (inv_uncertainty * m.squeeze(-1).squeeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        weights = inv_uncertainty / (weight_sum + self.eps)
        
        # Weighted fusion
        weighted_feats = feats_stack * weights
        fused = weighted_feats.sum(dim=1)  # [B, C, D, H, W]
        
        # Post-fusion scaling and bias
        fused = fused * self.post_scale + self.post_bias
        
        return fused


# Factory function for easy integration
def create_lightweight_fusion(
    fusion_type: str,
    channels: int,
    num_modalities: int | None = None,
    use_gamma: bool = True,
    use_null_token: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create lightweight fusion modules.
    
    Args:
        fusion_type: One of "learnable_weighted", "channel_gated", "uncertainty_weighted"
        channels: Number of feature channels
        num_modalities: Number of global modality groups
        use_gamma: Whether to use per-modality gamma scaling
        use_null_token: Whether to use null tokens for missing modalities
        **kwargs: Additional arguments passed to specific fusion classes
        
    Returns:
        Fusion module instance
    """
    fusion_map = {
        "learnable_weighted": LearnableWeightedFusion3D,
        "channel_gated": ChannelGatedFusion3D,
        "uncertainty_weighted": UncertaintyWeightedFusion3D
    }
    
    if fusion_type not in fusion_map:
        raise ValueError(f"Unknown fusion_type: {fusion_type}. Available: {list(fusion_map.keys())}")
    
    fusion_class = fusion_map[fusion_type]
    return fusion_class(
        channels=channels,
        num_modalities=num_modalities,
        use_gamma=use_gamma,
        use_null_token=use_null_token,
        **kwargs
    )
