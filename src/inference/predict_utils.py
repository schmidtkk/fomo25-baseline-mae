from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np


def parse_comma_separated_floats(s: Optional[str]) -> Optional[List[float]]:
    if s is None or str(s).strip() == "":
        return None
    return [float(x) for x in str(s).split(",")]


def average_scalar_probabilities(probabilities: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    if len(probabilities) == 0:
        raise ValueError("probabilities must be non-empty")
    probs = np.asarray(probabilities, dtype=np.float64)
    if weights is None:
        return float(probs.mean())
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != probs.shape[0]:
        raise ValueError("weights length must equal probabilities length")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("weights sum must be positive")
    w = w / s
    return float((w * probs).sum())


def apply_temperature_to_prob(prob: float, temperature: float) -> float:
    eps = 1e-6
    p = float(np.clip(prob, eps, 1.0 - eps))
    T = float(max(temperature, eps))
    logit = np.log(p) - np.log(1.0 - p)
    return float(1.0 / (1.0 + np.exp(-logit / T)))


