import torch
import torch.nn as nn



class MaskedMeanFusion3D(nn.Module):
    """
    Masked mean fusion for 3D feature maps across a variable number of modalities.

    For each scale, expects a stack of modality features with identical shape [B, C, D, H, W]
    and a presence mask [B, M]. Performs per-modality pre-alignment (GN(1,C) -> 1x1x1 Conv -> RMSNorm),
    optional per-modality channel-wise scaling, masked mean with safe count clamp, and learnable
    post-fusion scale/bias.
    """

    def __init__(self, channels: int, num_modalities: int | None = None, use_gamma: bool = True, use_null_token: bool = False):
        super().__init__()
        self.channels = channels
        self.pre_gn = nn.GroupNorm(1, channels, affine=True)
        self.pre_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)

        self.use_gamma = use_gamma and (num_modalities is not None)
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.ones(num_modalities, channels))

        self.use_null = use_null_token
        if self.use_null:
            self.null_token = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

        self.post_scale = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.post_bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.eps = 1e-6

    def _pre_align(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_gn(x)
        x = self.pre_proj(x)
        rms = x.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).add(self.eps).sqrt()
        return x / rms

    def forward(self, feats_list: list[torch.Tensor], mask: torch.Tensor, modality_ids: list[int] | None = None) -> torch.Tensor:
        """
        Args:
            feats_list: list of [B,C,D,H,W], one per present modality (length M)
            mask: [B,M] tensor, 1=present, 0=missing
            modality_ids: optional list of global modality indices (for per-modality gamma)
        Returns:
            fused: [B,C,D,H,W]
        """
        assert len(feats_list) > 0, "Empty feats_list provided to fusion"
        B, C, D, H, W = feats_list[0].shape
        M = len(feats_list)

        aligned = []
        for i, f in enumerate(feats_list):
            f = self._pre_align(f)
            if self.use_gamma:
                assert modality_ids is not None, "modality_ids required when use_gamma=True"
                gid = modality_ids[i]
                f = f * self.gamma[gid].view(1, C, 1, 1, 1)
            aligned.append(f)

        Fstack = torch.stack(aligned, dim=1)  # [B,M,C,D,H,W]
        # Ensure mask length matches modality count M
        if mask.dim() == 1:
            mask = mask.view(1, -1).expand(B, -1)
        if mask.shape[1] != M:
            if mask.shape[1] < M:
                pad = torch.zeros(B, M - mask.shape[1], dtype=mask.dtype, device=mask.device)
                mask = torch.cat([mask, pad], dim=1)
            else:
                mask = mask[:, :M]
        m = mask.view(B, M, 1, 1, 1, 1).type_as(Fstack)
        Fstack = Fstack * m

        if self.use_null:
            Fstack = Fstack + (1 - m) * self.null_token

        cnt = mask.sum(dim=1, keepdim=True).clamp_min(1.0).view(B, 1, 1, 1, 1, 1)
        fused = Fstack.sum(dim=1) / cnt  # [B,C,D,H,W]

        fused = fused * self.post_scale + self.post_bias
        return fused


