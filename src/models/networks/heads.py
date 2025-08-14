import torch
import torch.nn as nn


class ClsRegHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_p: float = 0.0):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Deterministic head: match encoder bottleneck channels
        self.dropout = nn.Dropout(p=float(dropout_p)) if dropout_p and dropout_p > 0 else nn.Identity()
        self.fc = nn.Linear(int(in_channels), int(num_classes))

    def forward(self, x):
        x = x[-1]  # only use bottleneck repr
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        # Guard against prior LazyLinear init with different channel count
        in_feat = x.shape[1]
        # nn.LazyLinear has attribute in_features after first init
        if hasattr(self.fc, "in_features") and self.fc.in_features is not None and self.fc.in_features != in_feat:
            out_feat = self.fc.out_features
            # Recreate a matching Linear on the correct device/dtype
            new_fc = nn.Linear(in_feat, out_feat, bias=True, device=x.device, dtype=x.dtype)
            self.fc = new_fc
        x = self.fc(x)
        return x
