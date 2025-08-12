from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torchmetrics.classification import AUROC


def average_probabilities(
    probabilities_per_model: Sequence[np.ndarray] | Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Average per-sample class probabilities across models/folds.

    Args:
        probabilities_per_model: Sequence of arrays/tensors with identical shape (N, C) or (N,).
            If 1D, interpreted as binary positive-class probabilities.
        weights: Optional weights for each model; will be normalized to sum to 1.

    Returns:
        np.ndarray with the same shape as inputs containing the averaged probabilities.
    """
    if len(probabilities_per_model) == 0:
        raise ValueError("probabilities_per_model must be non-empty")

    arrays = [
        x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        for x in probabilities_per_model
    ]

    first_shape = arrays[0].shape
    if any(arr.shape != first_shape for arr in arrays[1:]):
        raise ValueError("All probability arrays must have the same shape to average")

    if weights is None:
        weights_arr = np.ones(len(arrays), dtype=np.float64) / float(len(arrays))
    else:
        if len(weights) != len(arrays):
            raise ValueError("weights length must match number of arrays")
        weights_arr = np.asarray(weights, dtype=np.float64)
        total = weights_arr.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        weights_arr = weights_arr / total

    stacked = np.stack(arrays, axis=0)  # (M, ...)
    # Weighted average along model axis
    averaged = np.tensordot(weights_arr, stacked, axes=(0, 0))
    return averaged


def average_subject_probabilities(
    fold_subject_to_prob: Sequence[Mapping[str, float]],
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Average subject-level positive-class probabilities from multiple folds/models.

    Args:
        fold_subject_to_prob: List of mappings from subject_id -> prob_pos.
            All mappings must cover the same subject_id keys.
        weights: Optional weights per fold; normalized to sum to 1.

    Returns:
        Dict mapping subject_id -> averaged prob_pos.
    """
    if len(fold_subject_to_prob) == 0:
        return {}

    # Enforce identical subject sets to avoid silent drops
    subject_sets = [set(m.keys()) for m in fold_subject_to_prob]
    base_subjects = subject_sets[0]
    for s in subject_sets[1:]:
        if s != base_subjects:
            raise ValueError("All folds must contain identical subject_id keys for averaging")

    if weights is None:
        weights_arr = np.ones(len(fold_subject_to_prob), dtype=np.float64) / float(
            len(fold_subject_to_prob)
        )
    else:
        if len(weights) != len(fold_subject_to_prob):
            raise ValueError("weights length must match number of folds")
        weights_arr = np.asarray(weights, dtype=np.float64)
        total = weights_arr.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        weights_arr = weights_arr / total

    averaged: Dict[str, float] = {}
    for sid in sorted(base_subjects):
        probs = np.array([float(m[sid]) for m in fold_subject_to_prob], dtype=np.float64)
        averaged[sid] = float(np.dot(weights_arr, probs))
    return averaged


def auroc_from_subject_probabilities(
    subject_to_prob: Mapping[str, float], subject_to_target: Mapping[str, int]
) -> float:
    """
    Compute AUROC given per-subject positive-class probabilities and integer targets {0,1}.
    """
    subjects = sorted(subject_to_prob.keys())
    probs = torch.tensor([subject_to_prob[s] for s in subjects], dtype=torch.float32)
    targets = torch.tensor([subject_to_target[s] for s in subjects], dtype=torch.int64)

    if torch.unique(targets).numel() < 2:
        return float("nan")

    metric = AUROC(task="binary")
    return float(metric(probs, targets).item())


