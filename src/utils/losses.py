from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # BCE with logits to get base loss components
        prob = torch.sigmoid(logits)
        # p_t = p if y=1 else (1-p)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        # alpha_t = alpha if y=1 else (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # Focal loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiClassFocalLoss(torch.nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B] int
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        ce = F.nll_loss(log_probs, targets.long(), weight=self.alpha, reduction="none")
        focal = (1 - p_t) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


