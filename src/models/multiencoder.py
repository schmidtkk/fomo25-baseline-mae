from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn

from models.fusion.masked_mean import MaskedMeanFusion3D


class MultiModalEncoderWithFusion(nn.Module):
    """
    Wraps multiple modality-specific encoders and fuses their multi-scale features
    using MaskedMeanFusion3D at each scale. Returns a list [x0, x1, x2, x3, x4]
    compatible with existing UNet decoders/heads.
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
    ) -> None:
        super().__init__()
        self.modality_names = modality_names
        self.num_modalities = len(modality_names)
        self.global_vocab = global_vocab or ["t1", "t2", "flair", "dwi", "other"]

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

        # Build fusion layers per scale
        self.fusions = nn.ModuleList(
            [
                MaskedMeanFusion3D(
                    channels=c,
                    # Gamma is per global group; ensure parameter count equals len(global_vocab)
                    num_modalities=len(self.global_vocab),
                    use_gamma=use_gamma,
                    use_null_token=use_null_token,
                )
                for c in self.scale_channels
            ]
        )

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
        # Provide default modality_ids as global group ids per finetune modality if not supplied
        if modality_ids is None:
            modality_ids = self.modality_group_ids

        # Collect per-modality encoder outputs
        per_mod_feats: Dict[str, List[torch.Tensor]] = {}
        for i, name in enumerate(self.modality_names):
            feats = self.encoders[name](xs[name])  # list of 5 tensors
            per_mod_feats[name] = feats

        fused: List[torch.Tensor] = []
        for scale_idx in range(5):
            feats_list = [per_mod_feats[name][scale_idx] for name in self.modality_names]
            fused_s = self.fusions[scale_idx](feats_list, mask=mask, modality_ids=modality_ids)
            fused.append(fused_s)

        return fused


