from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn

from models.fusion.masked_mean import MaskedMeanFusion3D
from models.fusion.attention_fusion import AttentionFusion3D
from models.fusion.lightweight_fusion import create_lightweight_fusion


class MultiModalEncoderWithFusion(nn.Module):
    """
    Wraps multiple modality-specific encoders and fuses their multi-scale features
    using MaskedMeanFusion3D at each scale. Returns a list [x0, x1, x2, x3, x4]
    compatible with existing UNet decoders/heads.
    
    Supports modality-wise enable/disable switches for debugging and ablation studies.
    """

    def __init__(
        self,
        modality_names: List[str],
        encoder_factory: Callable[[], nn.Module],
        starting_filters: int = 64,
        num_modalities_global: Optional[int] = None,
        use_gamma: bool = True,
        use_null_token: bool = False,
        modality_to_global_group: Optional[Dict[str, str]] = None,
        global_vocab: Optional[List[str]] = None,
        enabled_modalities: Optional[List[str]] = None,
        fusion_type: str = "masked_mean",
    ) -> None:
        super().__init__()
        self.modality_names = modality_names
        self.num_modalities = len(modality_names)
        self.global_vocab = global_vocab or ["t1", "t2", "flair", "dwi", "other"]
        self.fusion_type = fusion_type
        
        # Modality enable/disable switches
        if enabled_modalities is None:
            self.enabled_modalities = set(modality_names)  # All enabled by default
        else:
            # Validate enabled modalities
            invalid = set(enabled_modalities) - set(modality_names)
            if invalid:
                raise ValueError(f"Invalid enabled modalities: {invalid}. Available: {modality_names}")
            self.enabled_modalities = set(enabled_modalities)
            
        print(f"🔧 Multi-encoder setup: {len(modality_names)} total, {len(self.enabled_modalities)} enabled")
        print(f"   Enabled: {sorted(self.enabled_modalities)}")
        print(f"   Disabled: {sorted(set(modality_names) - self.enabled_modalities)}")
        print(f"   Fusion type: {fusion_type}")

        def _infer_group(mod_name: str) -> str:
            n = mod_name.lower()
            if "adc" in n or "dwi" in n:
                return "dwi"
            if "flair" in n:
                return "flair"
            if n == "t1" or "t1w" in n:
                return "t1"
            if n == "t2" or "t2w" in n:
                return "t2"
            if any(tok in n for tok in ["swi", "t2s", "t2star", "t2*", "suscept"]):
                return "other"
            return "other"

        # Build mapping with heuristics if not provided or incomplete
        mtg: Dict[str, str] = {}
        if modality_to_global_group is not None:
            # normalize to lowercase tokens present in global vocab
            for k, v in modality_to_global_group.items():
                mtg[k] = v.lower()
        for m in modality_names:
            if m not in mtg:
                mtg[m] = _infer_group(m)
        self.modality_to_global_group = mtg

        # Build group ids aligned with modality_names
        group_index = {g: i for i, g in enumerate(self.global_vocab)}
        self.modality_group_ids = [group_index.get(self.modality_to_global_group[m], group_index["other"]) for m in self.modality_names]

        # Build an encoder per modality using provided factory (must create input_channels=1 encoders)
        encoders: Dict[str, nn.Module] = {}
        for name in modality_names:
            encoders[name] = encoder_factory()
        self.encoders = nn.ModuleDict(encoders)

        # Channels per scale per UNetEncoder
        self.scale_channels = [
            starting_filters * m for m in [1, 2, 4, 8, 16]
        ]

        # Build fusion layers per scale with selected fusion type
        fusion_class = MaskedMeanFusion3D
        fusion_kwargs = {
            "num_modalities": len(self.global_vocab),
            "use_gamma": use_gamma,
            "use_null_token": use_null_token,
        }
        
        # Support lightweight fusion methods
        if fusion_type in ["learnable_weighted", "channel_gated", "uncertainty_weighted"]:
            # Use lightweight fusion factory
            self.fusions = nn.ModuleList([
                create_lightweight_fusion(
                    fusion_type=fusion_type,
                    channels=c,
                    num_modalities=len(self.global_vocab),
                    use_gamma=use_gamma,
                    use_null_token=use_null_token
                )
                for c in self.scale_channels
            ])
        elif fusion_type in ["attention", "channel_attention", "spatial_attention", "hybrid_attention"]:
            fusion_class = AttentionFusion3D
            # Map fusion_type to attention_type parameter
            attention_type_map = {
                "attention": "channel",  # Default attention
                "channel_attention": "channel",
                "spatial_attention": "spatial", 
                "hybrid_attention": "hybrid"
            }
            fusion_kwargs["attention_type"] = attention_type_map[fusion_type]
            
            self.fusions = nn.ModuleList(
                [
                    fusion_class(
                        channels=c,
                        **fusion_kwargs
                    )
                    for c in self.scale_channels
                ]
            )
        else:
            # Default to masked_mean
            self.fusions = nn.ModuleList(
                [
                    fusion_class(
                        channels=c,
                        **fusion_kwargs
                    )
                    for c in self.scale_channels
                ]
            )

    def set_enabled_modalities(self, enabled_modalities: List[str]) -> None:
        """
        Dynamically enable/disable modalities for ablation studies.
        
        Args:
            enabled_modalities: List of modality names to enable
        """
        invalid = set(enabled_modalities) - set(self.modality_names)
        if invalid:
            raise ValueError(f"Invalid modalities: {invalid}. Available: {self.modality_names}")
        
        old_enabled = self.enabled_modalities.copy()
        self.enabled_modalities = set(enabled_modalities)
        
        print(f"🔄 Modality switches updated:")
        print(f"   Previously enabled: {sorted(old_enabled)}")
        print(f"   Now enabled: {sorted(self.enabled_modalities)}")
        
        # Update fusion layers to handle the new modality configuration
        self._update_fusion_masks()
    
    def _update_fusion_masks(self) -> None:
        """Update fusion layer configurations for current enabled modalities."""
        # This will be called during forward pass - fusion layers adapt automatically
        pass
    
    def get_modality_status(self) -> Dict[str, bool]:
        """Get current enable/disable status of all modalities."""
        return {mod: mod in self.enabled_modalities for mod in self.modality_names}
    
    def enable_modality(self, modality_name: str) -> None:
        """Enable a specific modality."""
        if modality_name not in self.modality_names:
            raise ValueError(f"Unknown modality: {modality_name}")
        self.enabled_modalities.add(modality_name)
        print(f"✅ Enabled modality: {modality_name}")
    
    def disable_modality(self, modality_name: str) -> None:
        """Disable a specific modality."""
        if modality_name not in self.modality_names:
            raise ValueError(f"Unknown modality: {modality_name}")
        if len(self.enabled_modalities) <= 1:
            raise ValueError("Cannot disable modality - at least one must remain enabled")
        self.enabled_modalities.discard(modality_name)
        print(f"❌ Disabled modality: {modality_name}")

    @torch.no_grad()
    def _split_modalities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Split input [B,M,D,H,W] into a dict of [B,1,D,H,W] per modality.
        """
        assert x.dim() == 5, f"Expected 5D tensor, got {x.shape}"
        B, M, D, H, W = x.shape
        assert M == self.num_modalities, f"Expected {self.num_modalities} modalities but got {M}"
        return {name: x[:, i : i + 1] for i, name in enumerate(self.modality_names)}

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, modality_ids: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Args:
            x: [B,M,D,H,W] input volume
            mask: [B,M] presence mask (1=present). If None, will be derived from x!=0.
            modality_ids: optional list[int] mapping each order position to a global modality id for per-modality gamma.
        Returns:
            List of 5 fused feature maps [B,Cs,D,H,W] per scale.
        """
        B, M, D, H, W = x.shape
        if mask is None:
            with torch.no_grad():
                mask = (x.abs().view(B, M, -1).sum(dim=-1) > 0).to(x.dtype)

        xs = self._split_modalities(x)
        
        # Filter for only enabled modalities
        enabled_indices = []
        enabled_modality_names = []
        enabled_xs = {}
        enabled_mask = []
        
        for i, name in enumerate(self.modality_names):
            if name in self.enabled_modalities:
                enabled_indices.append(i)
                enabled_modality_names.append(name)
                enabled_xs[name] = xs[name]
                enabled_mask.append(mask[:, i] if mask.shape[1] > i else torch.ones(B, device=mask.device))
        
        if len(enabled_modality_names) == 0:
            raise RuntimeError("No modalities are enabled!")
        
        # Convert enabled_mask to tensor [B, enabled_M]
        enabled_mask_tensor = torch.stack(enabled_mask, dim=1)
        
        # Provide default modality_ids as global group ids per enabled modality if not supplied
        if modality_ids is None:
            enabled_modality_ids = [self.modality_group_ids[i] for i in enabled_indices]
        else:
            enabled_modality_ids = [modality_ids[i] for i in enabled_indices]

        # Collect per-modality encoder outputs for enabled modalities only
        per_mod_feats: Dict[str, List[torch.Tensor]] = {}
        for name in enabled_modality_names:
            with torch.set_grad_enabled(name in self.enabled_modalities):
                feats = self.encoders[name](enabled_xs[name])  # list of 5 tensors
                per_mod_feats[name] = feats

        fused: List[torch.Tensor] = []
        for scale_idx in range(5):
            feats_list = [per_mod_feats[name][scale_idx] for name in enabled_modality_names]
            fused_s = self.fusions[scale_idx](
                feats_list, 
                mask=enabled_mask_tensor, 
                modality_ids=enabled_modality_ids
            )
            fused.append(fused_s)

        return fused


