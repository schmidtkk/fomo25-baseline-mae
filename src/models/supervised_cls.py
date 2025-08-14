from typing import Optional, Dict, List
import os
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC

from models.supervised_base import BaseSupervisedModel


class SupervisedClsModel(BaseSupervisedModel):
    """
    Supervised model for classification tasks.
    Inherits from BaseSupervisedModel and implements classification-specific functionality.
    """

    def __init__(
        self,
        config: dict = {},
        learning_rate: float = 1e-3,
        do_compile: Optional[bool] = False,
        compile_mode: Optional[str] = "default",
        weight_decay: float = 3e-5,
        amsgrad: bool = False,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
    ):
        super().__init__(
            config=config,
            learning_rate=learning_rate,
            do_compile=do_compile,
            compile_mode=compile_mode,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            eps=eps,
            betas=betas,
            deep_supervision=False,  # Classification doesn't use deep supervision
        )

    def _configure_metrics(self, prefix: str):
        """
        Configure classification-specific metrics

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of classification metrics
        """
        return MetricCollection(
            {
                f"{prefix}/accuracy": Accuracy(
                    task="multiclass", num_classes=self.num_classes
                ),
                f"{prefix}/precision": Precision(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
                f"{prefix}/recall": Recall(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
                f"{prefix}/f1": F1Score(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
            }
        )

    def _configure_losses(self):
        """
        Configure classification-specific loss functions

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """
        # For classification, use cross-entropy with optional label smoothing
        smoothing = float(self.config.get("label_smoothing", 0.0))
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
        return loss_fn, loss_fn

    # ---- Subject-level AUROC aggregation (Task 1) ----
    def on_validation_epoch_start(self):
        # subject_id -> [sum_prob_pos, count, target]
        self._val_subject_aggr: Dict[str, List[float]] = {}
        # accumulate scalar metrics to average at epoch end (robustness)
        self._val_loss_sum = 0.0
        self._val_count = 0
        # deterministic flip patterns for TTA
        self._val_tta_enable = bool(self.config.get("val_tta_enable", False))
        self._val_tta_views = int(self.config.get("val_tta_views", 1))
        # Precompute flip codes (up to 8: none, x, y, z, xy, xz, yz, xyz)
        self._tta_codes = [
            (), (2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)
        ][: max(1, min(8, self._val_tta_views))]
        # deterministic translation offsets (center + axis shifts)
        self._val_tta_offsets = int(self.config.get("val_tta_offsets", 1))
        self._val_tta_offset_frac = float(self.config.get("val_tta_offset_frac", 0.25))
        # Build offsets: center, +/- along each axis (up to 7 total)
        base = [(0, 0, 0)]
        shifts = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        self._tta_offsets = base + shifts
        self._tta_offsets = self._tta_offsets[: max(1, min(7, self._val_tta_offsets))]

    @staticmethod
    def _extract_subject_id(path_str: str) -> str:
        # Use folder name as subject ID (works for fusion); fallback to basename
        return os.path.basename(path_str.rstrip("/"))

    @staticmethod
    def _mean_probs_targets_from_aggr(aggr: Dict[str, List[float]]):
        probs = []
        targets = []
        for _, (sum_prob, cnt, tgt) in aggr.items():
            if cnt > 0:
                probs.append(sum_prob / cnt)
                targets.append(tgt)
        if len(probs) == 0:
            return None, None
        return torch.tensor(probs, dtype=torch.float32), torch.tensor(targets, dtype=torch.int64)

    def validation_step(self, batch, _batch_idx):
        # Run base validation logging (loss + per-sample metrics)
        inputs, target, file_path = self._process_batch(batch)
        if not self._val_tta_enable or (len(self._tta_codes) == 1 and len(self._tta_offsets) == 1):
            output = self(inputs)
        else:
            B, M, D, H, W = inputs.shape
            dz = int(round(self._val_tta_offset_frac * D))
            dy = int(round(self._val_tta_offset_frac * H))
            dx = int(round(self._val_tta_offset_frac * W))
            # Average logits over deterministic offsets and flips
            logits_sum = None
            num = 0
            for oz, oy, ox in self._tta_offsets:
                # translate via roll; for classification logits global pooled, this is acceptable
                xoff = torch.roll(inputs, shifts=(oz * dz, oy * dy, ox * dx), dims=(2, 3, 4))
                for code in self._tta_codes:
                    x = xoff
                    if len(code) > 0:
                        x = torch.flip(x, dims=list(code))
                    logits = self(x)
                    logits_sum = logits if logits_sum is None else (logits_sum + logits)
                    num += 1
            output = logits_sum / float(max(1, num))
        loss = self.loss_fn_val(output, target)
        metrics = self.compute_metrics(self.val_metrics, output, target)
        # accumulate
        self._val_loss_sum += float(loss.detach().cpu())
        self._val_count += 1
        # still log per-step for live feedback
        self.log_dict({"val/loss": loss} | metrics, prog_bar=False, logger=True)

        # Additional subject-level aggregation for binary AUROC in Task 1
        if self.num_classes == 2:
            # reuse computed output above
            prob = F.softmax(output, dim=1)[:, 1].detach().cpu()
            target = target.detach().cpu()

            # file_path may be a list/tuple of strings or a single string
            if isinstance(file_path, (list, tuple)):
                paths = list(file_path)
            else:
                paths = [file_path] * prob.shape[0]

            for i in range(prob.shape[0]):
                sid = self._extract_subject_id(str(paths[i]))
                p = float(prob[i].item())
                t = int(target[i].item())
                if sid not in self._val_subject_aggr:
                    self._val_subject_aggr[sid] = [0.0, 0.0, t]
                self._val_subject_aggr[sid][0] += p
                self._val_subject_aggr[sid][1] += 1.0

    def on_validation_epoch_end(self):
        # Compute subject-level AUROC if binary classification
        if self.num_classes == 2 and hasattr(self, "_val_subject_aggr"):
            probs, targets = self._mean_probs_targets_from_aggr(self._val_subject_aggr)
            if probs is not None and targets is not None:
                # Ensure both classes are present
                if torch.unique(targets).numel() >= 2:
                    auroc_metric = AUROC(task="binary")
                    auroc_val = auroc_metric(probs, targets)
                    self.log("val/auroc_subject", auroc_val, prog_bar=True, logger=True)
                else:
                    # Insufficient class variety; skip AUROC
                    self.log("val/auroc_subject", torch.nan, prog_bar=True, logger=True)

            # Export per-subject probabilities/targets for analysis
            try:
                import csv, os
                out_dir = os.path.join(self.config.get("version_dir", "."))
                os.makedirs(out_dir, exist_ok=True)
                csv_path = os.path.join(out_dir, f"val_subject_scores_epoch_{self.current_epoch}.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["subject_id", "mean_prob_pos", "target"])
                    for sid, (sum_prob, cnt, tgt) in sorted(self._val_subject_aggr.items()):
                        mp = (sum_prob / max(cnt, 1.0)) if cnt > 0 else float("nan")
                        writer.writerow([sid, mp, int(tgt)])
            except Exception:
                pass

        # Log averaged validation loss/metrics
        if getattr(self, "_val_count", 0) > 0:
            avg_loss = torch.tensor(self._val_loss_sum / max(self._val_count, 1), dtype=torch.float32, device=self.device)
            self.log("val/loss_epoch", avg_loss, prog_bar=True, logger=True)

    def _process_batch(self, batch):
        """
        Process classification batch data

        Args:
            batch: Input batch

        Returns:
            tuple: (inputs, target, file_path)
        """
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        # Convert target to long for classification tasks
        target = target.long()

        # Only squeeze if dimension exists
        if target.dim() > 1:
            target = target.squeeze(1)

        return inputs, target, file_path

    def compute_metrics(self, metrics, output, target, ignore_index=None):
        """
        Compute classification metrics

        Args:
            metrics: Metrics collection
            output: Model output
            target: Ground truth
            ignore_index: Index to ignore in metrics (not used in classification)

        Returns:
            dict: Dictionary of computed metrics
        """
        # Use the same approach for binary and multi-class classification
        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)
        return metrics(probabilities, target)
