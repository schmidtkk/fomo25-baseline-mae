from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


class TemperatureScaler:
    """
    Platt-style temperature scaling for binary classification.
    Optimizes a single scalar temperature T > 0 to minimize NLL on a calibration set.
    """

    def __init__(self, init_temperature: float = 1.0):
        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive")
        self.log_temperature = torch.nn.Parameter(torch.tensor(math.log(init_temperature), dtype=torch.float32))

    @staticmethod
    def _probs_to_logits(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.clamp(probs, eps, 1.0 - eps)
        return torch.log(probs) - torch.log(1.0 - probs)

    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward_logits(self, logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature()
        return logits / T

    def forward_probs(self, probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        logits = self._probs_to_logits(probs, eps=eps)
        return torch.sigmoid(self.forward_logits(logits))

    def fit_from_probs(
        self, probs: torch.Tensor, targets: torch.Tensor, lr: float = 0.05, steps: int = 300, eps: float = 1e-6
    ) -> float:
        """
        Fit temperature using given probabilities and binary targets.
        Returns the learned temperature value.
        """
        probs = probs.detach().float()
        targets = targets.detach().float()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

        bce = torch.nn.BCEWithLogitsLoss()

        logits = self._probs_to_logits(probs, eps=eps)

        def closure():
            optimizer.zero_grad(set_to_none=True)
            scaled = self.forward_logits(logits)
            loss = bce(scaled, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature().item())

    @staticmethod
    def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
        probs = probs.detach().float()
        targets = targets.detach().float()
        return float(torch.mean((probs - targets) ** 2).item())


