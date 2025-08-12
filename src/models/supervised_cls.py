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
        # For classification, we typically use cross-entropy loss
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn, loss_fn

    # ---- Subject-level AUROC aggregation (Task 1) ----
    def on_validation_epoch_start(self):
        # subject_id -> [sum_prob_pos, count, target]
        self._val_subject_aggr: Dict[str, List[float]] = {}

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
        super().validation_step(batch, _batch_idx)

        # Additional subject-level aggregation for binary AUROC in Task 1
        if self.num_classes == 2:
            inputs, target, file_path = self._process_batch(batch)
            output = self(inputs)
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
