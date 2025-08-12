from __future__ import annotations

from typing import Dict, Optional

import torch
from lightning.pytorch.callbacks import Callback
import os
from torch.nn.parameter import UninitializedParameter


class ModelEMACallback(Callback):
    """
    Exponential Moving Average (EMA) of model parameters.

    - Keeps an EMA copy of the model parameters and buffers (floating dtypes only)
    - Updates EMA after each train batch
    - Swaps EMA weights in for validation; restores original after
    """

    def __init__(
        self,
        decay: float = 0.999,
        device: str | None = None,
        save_ema_best: bool = True,
        monitor: Optional[str] = None,
        mode: str = "max",
        filename: str = "ema-best",
    ):
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.device = device
        self._ema_state: Dict[str, torch.Tensor] = {}
        self._backup_state: Dict[str, torch.Tensor] = {}
        # Saving config
        self.save_ema_best = bool(save_ema_best)
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self._best_score: Optional[float] = None

    @torch.no_grad()
    def _clone_float_state(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in state_dict.items():
            # Skip lazy/uninitialized params
            if isinstance(v, UninitializedParameter):
                continue
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                out[k] = v.detach().clone()
        return out

    @torch.no_grad()
    def on_fit_start(self, trainer, pl_module):
        # Try to initialize EMA with the model's initial state; if lazy params present, defer to first batch
        state = pl_module.state_dict()
        try:
            self._ema_state = self._clone_float_state(state)
        except ValueError:
            # Defer initialization until parameters are materialized
            self._ema_state = {}
        if self.device is not None and self._ema_state:
            for k in list(self._ema_state.keys()):
                self._ema_state[k] = self._ema_state[k].to(self.device)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update EMA: ema = decay * ema + (1 - decay) * param
        current = pl_module.state_dict()
        # If EMA was not initialized at fit start (due to lazy params), initialize now
        if not self._ema_state:
            self._ema_state = self._clone_float_state(current)
            if self.device is not None and self._ema_state:
                for k in list(self._ema_state.keys()):
                    self._ema_state[k] = self._ema_state[k].to(self.device)
            return
        for k, v in current.items():
            if k in self._ema_state and isinstance(v, torch.Tensor) and v.is_floating_point() and not isinstance(v, UninitializedParameter):
                src = v.detach()
                if self.device is not None:
                    src = src.to(self.device)
                self._ema_state[k].mul_(self.decay).add_(src, alpha=1.0 - self.decay)

    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module):
        # Backup current state and swap in EMA for validation
        if not self._ema_state:
            return
        state = pl_module.state_dict()
        self._backup_state = self._clone_float_state(state)
        # Build a new state dict with EMA values where available
        new_state = {}
        for k, v in state.items():
            if k in self._ema_state:
                ema_v = self._ema_state[k]
                # Move to module device if needed
                ema_v = ema_v.to(v.device)
                new_state[k] = ema_v
            else:
                new_state[k] = v
        pl_module.load_state_dict({**state, **new_state}, strict=False)

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        # At this point, EMA weights are still loaded in the module.
        # Optionally save EMA checkpoint if metric improved.
        if self.save_ema_best and self.monitor is not None:
            metric_val = trainer.callback_metrics.get(self.monitor)
            try:
                current_score = float(metric_val)
            except Exception:
                current_score = None
            if current_score is not None:
                is_better = (
                    (self._best_score is None)
                    or (self.mode == "max" and current_score > self._best_score)
                    or (self.mode == "min" and current_score < self._best_score)
                )
                if is_better:
                    self._best_score = current_score
                    # Determine checkpoint directory from the main checkpoint callback
                    ckpt_cb = getattr(trainer, "checkpoint_callback", None)
                    dirpath = None
                    if ckpt_cb is not None:
                        dirpath = getattr(ckpt_cb, "dirpath", None)
                    if not dirpath:
                        # Fallback to logger dir if available
                        log_dir = getattr(getattr(trainer, "logger", None), "log_dir", None)
                        dirpath = os.path.join(log_dir, "checkpoints") if log_dir else None
                    if dirpath:
                        os.makedirs(dirpath, exist_ok=True)
                        save_path = os.path.join(dirpath, f"{self.filename}.ckpt")
                        # Build a minimal Lightning checkpoint with EMA weights
                        checkpoint = {
                            "state_dict": pl_module.state_dict(),
                            "epoch": getattr(trainer, "current_epoch", None),
                            "global_step": getattr(trainer, "global_step", None),
                            "monitor": self.monitor,
                            "score": self._best_score,
                        }
                        torch.save(checkpoint, save_path)
        # Restore original (non-EMA) weights after optional save
        if self._backup_state:
            current = pl_module.state_dict()
            restore = {}
            for k, v in current.items():
                if k in self._backup_state:
                    b = self._backup_state[k]
                    restore[k] = b.to(v.device)
                else:
                    restore[k] = v
            pl_module.load_state_dict({**current, **restore}, strict=False)
            self._backup_state = {}


