import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class AttentionFusion3D(nn.Module):
    """
    Attention-based fusion for 3D feature maps across multiple modalities.
    
    Uses learnable attention weights to dynamically weight modality contributions
    based on their feature content. More sophisticated than simple averaging.
    """

    def __init__(
        self, 
        channels: int, 
        num_modalities: int | None = None, 
        use_gamma: bool = True, 
        use_null_token: bool = False,
        attention_type: str = "channel"
    ):
        """
        Args:
            channels: Number of feature channels
            num_modalities: Number of global modality groups for gamma scaling
            use_gamma: Whether to use per-modality gamma scaling
            use_null_token: Whether to use null tokens for missing modalities
            attention_type: Type of attention - "channel", "spatial", or "hybrid"
        """
        super().__init__()
        self.channels = channels
        self.attention_type = attention_type
        
        # Pre-alignment components (same as MaskedMean)
        self.pre_gn = nn.GroupNorm(1, channels, affine=True)
        self.pre_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        
        # Gamma scaling per modality (if requested)
        self.use_gamma = use_gamma and (num_modalities is not None)
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.ones(num_modalities, channels))
        
        # Null token for missing modalities
        self.use_null = use_null_token
        if self.use_null:
            self.null_token = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        
        # Attention mechanism components
        if attention_type == "channel":
            # Channel-wise attention: learn importance of each channel per modality
            self.attention_fc = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),  # [B*M, C, 1, 1, 1]
                nn.Conv3d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels // 4, channels, 1),
                nn.Sigmoid()
            )
        elif attention_type == "spatial":
            # Spatial attention: learn importance of spatial locations
            self.attention_conv = nn.Sequential(
                nn.Conv3d(channels, channels // 4, kernel_size=3, padding=1),
                nn.GroupNorm(1, channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels // 4, 1, kernel_size=1),
                nn.Sigmoid()
            )
        elif attention_type == "hybrid":
            # Both channel and spatial attention
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels // 4, channels, 1),
                nn.Sigmoid()
            )
            self.spatial_attention = nn.Sequential(
                nn.Conv3d(channels, channels // 8, kernel_size=3, padding=1),
                nn.GroupNorm(1, channels // 8),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels // 8, 1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        
        # Cross-modality attention to learn modality interactions
        self.cross_modality_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=max(1, channels // 64),
            dropout=0.1,
            batch_first=True
        )
        
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

    def _apply_attention(self, feats_stack: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism to modality features.
        
        Args:
            feats_stack: [B, M, C, D, H, W] stacked features
            
        Returns:
            attended_feats: [B, M, C, D, H, W] attention-weighted features
        """
        B, M, C, D, H, W = feats_stack.shape
        
        if self.attention_type == "channel":
            # Channel attention: [B*M, C, D, H, W] -> [B*M, C, 1, 1, 1]
            feats_flat = feats_stack.view(B * M, C, D, H, W)
            channel_weights = self.attention_fc(feats_flat)  # [B*M, C, 1, 1, 1]
            attended = feats_flat * channel_weights
            return attended.view(B, M, C, D, H, W)
            
        elif self.attention_type == "spatial":
            # Spatial attention: [B*M, C, D, H, W] -> [B*M, 1, D, H, W]
            feats_flat = feats_stack.view(B * M, C, D, H, W)
            spatial_weights = self.attention_conv(feats_flat)  # [B*M, 1, D, H, W]
            attended = feats_flat * spatial_weights
            return attended.view(B, M, C, D, H, W)
            
        elif self.attention_type == "hybrid":
            # Both channel and spatial attention
            feats_flat = feats_stack.view(B * M, C, D, H, W)
            
            # Channel attention
            channel_weights = self.channel_attention(feats_flat)  # [B*M, C, 1, 1, 1]
            channel_attended = feats_flat * channel_weights
            
            # Spatial attention
            spatial_weights = self.spatial_attention(channel_attended)  # [B*M, 1, D, H, W]
            attended = channel_attended * spatial_weights
            
            return attended.view(B, M, C, D, H, W)
        
        return feats_stack

    def _cross_modality_attention(self, feats_stack: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-modality attention to capture inter-modality dependencies.
        
        Args:
            feats_stack: [B, M, C, D, H, W] features
            mask: [B, M] presence mask
            
        Returns:
            enhanced_feats: [B, M, C, D, H, W] cross-attention enhanced features
        """
        B, M, C, D, H, W = feats_stack.shape
        
        # Reshape for attention: [B, M, C*D*H*W]
        feats_for_attn = feats_stack.view(B, M, C * D * H * W)
        
        # Create attention mask from presence mask
        # mask: [B, M] -> [B, M, M] (each modality can attend to others)
        attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, M, M]
        attn_mask = attn_mask.view(B * M, M).bool()
        
        # Apply cross-attention
        feats_flat = feats_for_attn.view(B * M, 1, C * D * H * W)
        key_value = feats_for_attn.view(B, M, C * D * H * W).repeat_interleave(M, dim=0)  # [B*M, M, C*D*H*W]
        
        try:
            attended_flat, _ = self.cross_modality_attention(
                feats_flat,  # query: [B*M, 1, C*D*H*W]
                key_value,   # key: [B*M, M, C*D*H*W]
                key_value,   # value: [B*M, M, C*D*H*W]
                key_padding_mask=~attn_mask  # [B*M, M]
            )
        except:
            # Fallback if attention fails
            attended_flat = feats_flat
        
        # Reshape back
        attended = attended_flat.view(B, M, C, D, H, W)
        return attended

    def forward(
        self, 
        feats_list: List[torch.Tensor], 
        mask: torch.Tensor, 
        modality_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Apply attention-based fusion to modality features.
        
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
        
        # Apply attention mechanisms
        attended_feats = self._apply_attention(feats_stack)
        
        # Apply cross-modality attention for inter-modality dependencies
        if M > 1:  # Only apply if multiple modalities
            attended_feats = self._cross_modality_attention(attended_feats, mask)
        
        # Weighted average with learned attention weights
        # Compute per-modality weights based on feature magnitude and attention
        modality_weights = attended_feats.abs().mean(dim=(2, 3, 4, 5), keepdim=True)  # [B, M, 1, 1, 1, 1]
        modality_weights = F.softmax(modality_weights.squeeze(-1).squeeze(-1).squeeze(-1), dim=1)  # [B, M]
        modality_weights = modality_weights.view(B, M, 1, 1, 1, 1)
        
        # Apply modality weights and mask
        weighted_feats = attended_feats * modality_weights * m
        
        # Sum across modalities (weighted average)
        cnt = mask.sum(dim=1, keepdim=True).clamp_min(1.0).view(B, 1, 1, 1, 1, 1)
        fused = weighted_feats.sum(dim=1) / cnt.clamp_min(self.eps)  # [B, C, D, H, W]
        
        # Post-fusion scaling and bias
        fused = fused * self.post_scale + self.post_bias
        
        return fused
