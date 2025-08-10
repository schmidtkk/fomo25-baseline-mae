import torch
import torch.nn as nn


class ClsRegHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Use LazyLinear to be robust to encoder channel width changes
        # in_channels is kept for API compatibility but not used directly.
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = x[-1]  # only use bottleneck repr
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
