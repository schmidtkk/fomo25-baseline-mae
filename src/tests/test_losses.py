import torch

from utils.losses import BinaryFocalLoss, MultiClassFocalLoss


def test_binary_focal_loss_runs_and_scales():
    logits = torch.tensor([0.0, 2.0, -1.0])
    targets = torch.tensor([0.0, 1.0, 0.0])
    loss = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    val = loss(logits, targets)
    assert torch.isfinite(val)


def test_multiclass_focal_loss_runs_and_scales():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.5, 1.5, -0.5]])
    targets = torch.tensor([0, 1])
    loss = MultiClassFocalLoss(gamma=2.0)
    val = loss(logits, targets)
    assert torch.isfinite(val)


