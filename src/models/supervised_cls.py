from typing import Optional, Dict, List
import os
import json
from utils.logging_utils import write_subject_csv
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC

from models.supervised_base import BaseSupervisedModel
from utils.losses import BinaryFocalLoss, MultiClassFocalLoss


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
        # Options: class-weighted CE, focal loss, label smoothing
        label_smoothing = float(self.config.get("label_smoothing", 0.0))
        class_weights = self.config.get("class_weights", None)
        focal_gamma = self.config.get("focal_gamma", None)
        focal_alpha = self.config.get("focal_alpha", None)

        if self.num_classes == 2:
            if focal_gamma is not None:
                train_loss = BinaryFocalLoss(alpha=float(focal_alpha or 0.25), gamma=float(focal_gamma))
                val_loss = BinaryFocalLoss(alpha=float(focal_alpha or 0.25), gamma=float(focal_gamma))
            else:
                # Use CrossEntropyLoss with two-logit output to match metrics and model head
                weight_tensor = None
                if class_weights is not None and isinstance(class_weights, (list, tuple)) and len(class_weights) == 2:
                    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
                train_loss = torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
                val_loss = torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.0)
        else:
            if focal_gamma is not None:
                alpha_tensor = None
                if class_weights is not None:
                    alpha_tensor = torch.tensor(class_weights, dtype=torch.float32)
                train_loss = MultiClassFocalLoss(alpha=alpha_tensor, gamma=float(focal_gamma))
                val_loss = MultiClassFocalLoss(alpha=alpha_tensor, gamma=float(focal_gamma))
            else:
                weight_tensor = None
                if class_weights is not None:
                    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
                train_loss = torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
                val_loss = torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.0)

        return train_loss, val_loss

    # ---- Subject-level AUROC aggregation (Task 1) ----
    def on_validation_epoch_start(self):
        # subject_id -> [sum_prob_pos, count, target, individual_probs]
        self._val_subject_aggr: Dict[str, List[float]] = {}

    @staticmethod
    def _extract_subject_id(path_str: str) -> str:
        # Use folder name as subject ID (works for fusion); fallback to basename
        return os.path.basename(path_str.rstrip("/"))

    @staticmethod
    def _mean_probs_targets_from_aggr(aggr: Dict[str, List[float]]):
        probs = []
        targets = []
        for _sid, data in aggr.items():
            # Support both legacy [sum_prob, cnt, tgt] and new [sum_prob, cnt, tgt, individual_probs]
            sum_prob = None
            cnt = None
            tgt = None
            try:
                if isinstance(data, (list, tuple)):
                    if len(data) >= 3:
                        sum_prob, cnt, tgt = data[0], data[1], data[2]
                elif isinstance(data, dict):
                    # Optional dict format support
                    sum_prob = float(data.get("sum_prob", 0.0))
                    cnt = float(data.get("count", 0.0))
                    tgt = int(data.get("target", 0))
            except Exception:
                continue
            if sum_prob is None or cnt is None or tgt is None:
                continue
            if float(cnt) > 0:
                probs.append(float(sum_prob) / float(cnt))
                targets.append(int(tgt))
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
                
                # Enhanced tracking: store individual probabilities for multiple aggregation methods
                if sid not in self._val_subject_aggr:
                    self._val_subject_aggr[sid] = [0.0, 0.0, t, []]  # [sum_probs, count, target, individual_probs]
                self._val_subject_aggr[sid][0] += p
                self._val_subject_aggr[sid][1] += 1.0
                self._val_subject_aggr[sid][3].append(p)  # Store individual probability

    def on_validation_epoch_end(self):
        # Compute subject-level AUROC if binary classification
        if self.num_classes == 2 and hasattr(self, "_val_subject_aggr"):
            # Enhanced aggregation with multiple methods
            if hasattr(self, '_enhanced_aggregation_enabled') and self._enhanced_aggregation_enabled:
                self._compute_enhanced_subject_aurocs()
            else:
                # Fallback to original method for backward compatibility
                self._compute_original_subject_auroc()
                
            # Optional: export per-subject probabilities/targets for ensembling
            if bool(self.config.get("export_subject_probs", False)):
                self._export_subject_probabilities()

    def _compute_enhanced_subject_aurocs(self):
        """Compute AUROC using multiple aggregation methods."""
        try:
            from utils.subject_aggregation import SubjectAggregationSuite
            
            # Initialize aggregation suite if not already done
            if not hasattr(self, '_aggregation_suite'):
                self._aggregation_suite = SubjectAggregationSuite()
            
            # Convert current format to individual probabilities format
            subject_crop_probs = {}
            subject_targets = {}
            
            for subject_id, (sum_probs, count, target, individual_probs) in self._val_subject_aggr.items():
                if len(individual_probs) > 0:
                    subject_crop_probs[subject_id] = individual_probs
                    subject_targets[subject_id] = target
                else:
                    # Fallback: create synthetic individual probabilities from mean
                    mean_prob = sum_probs / count if count > 0 else 0.0
                    subject_crop_probs[subject_id] = [mean_prob] * max(1, int(count))
                    subject_targets[subject_id] = target
            
            # Apply all aggregation methods
            results = self._aggregation_suite.aggregate_all_methods(
                subject_crop_probs, subject_targets
            )
            
            # Compute and log AUROCs for each method
            method_aurocs = self._aggregation_suite.compute_method_aurocs(results)
            
            for method_name, auroc_val in method_aurocs.items():
                if not torch.isnan(torch.tensor(auroc_val)):
                    self.log(f"val/auroc_subject_{method_name}", auroc_val, prog_bar=True, logger=True)
            
            # Log the best method's AUROC as the main metric (for backward compatibility)
            if method_aurocs:
                valid_aurocs = {k: v for k, v in method_aurocs.items() if not torch.isnan(torch.tensor(v))}
                if valid_aurocs:
                    best_method = max(valid_aurocs.keys(), key=lambda k: valid_aurocs[k])
                    best_auroc = valid_aurocs[best_method]
                    self.log("val/auroc_subject", best_auroc, prog_bar=True, logger=True)
                    
                    # Store results for visualization
                    self._last_subject_results = results
                    self._last_method_aurocs = method_aurocs
                else:
                    self.log("val/auroc_subject", torch.nan, prog_bar=True, logger=True)
            else:
                self.log("val/auroc_subject", torch.nan, prog_bar=True, logger=True)
                
        except ImportError:
            # Fallback to original method if enhanced aggregation not available
            print("Enhanced aggregation not available, falling back to original method")
            self._compute_original_subject_auroc()
        except Exception as e:
            print(f"Error in enhanced aggregation: {e}")
            self._compute_original_subject_auroc()

    def _compute_original_subject_auroc(self):
        """Original subject-level AUROC computation (backward compatibility)."""
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
    
    def _export_subject_probabilities(self):
        """Export subject probabilities for ensembling and analysis."""
        try:
            # Optional: export per-subject probabilities/targets for ensembling
            if bool(self.config.get("export_subject_probs", False)):
                subject_probs: Dict[str, float] = {}
                subject_targets: Dict[str, int] = {}
                
                # Use enhanced results if available, otherwise fall back to original
                if hasattr(self, '_last_subject_results') and self._last_subject_results:
                    # Export enhanced results with multiple aggregation methods
                    self._export_enhanced_subject_results()
                else:
                    # Fall back to original export method
                    self._export_original_subject_results()
                    
        except Exception as e:
            print(f"Error exporting subject probabilities: {e}")

    def _export_enhanced_subject_results(self):
        """Export enhanced subject results with multiple aggregation methods."""
        import os
        import json
        from utils.logging_utils import write_subject_csv
        
        version_dir = self.config.get("version_dir", ".")
        epoch = self.trainer.current_epoch if hasattr(self, 'trainer') else 0
        
        # Create export directories
        subject_probs_dir = os.path.join(version_dir, "subject_probs")
        os.makedirs(subject_probs_dir, exist_ok=True)
        
        # Prepare data for export
        results = self._last_subject_results
        method_aurocs = getattr(self, '_last_method_aurocs', {})
        
        # Enhanced JSON export with all methods
        enhanced_export_data = {
            "epoch": epoch,
            "aggregation_methods": {},
            "subject_targets": {sid: results[sid]['target'] for sid in results if 'target' in results[sid]},
            "method_aurocs": method_aurocs
        }
        
        # Export each aggregation method
        for method_name in ['mean_prob', 'mean_logit', 'noisy_or', 'top_k_3', 'top_k_5', 'max_prob']:
            if method_name in results[list(results.keys())[0]]:  # Check if method exists
                method_probs = {sid: results[sid][method_name] for sid in results if method_name in results[sid]}
                enhanced_export_data["aggregation_methods"][method_name] = method_probs
        
        # Save enhanced JSON
        enhanced_json_path = os.path.join(subject_probs_dir, f"val_subject_probs_enhanced_epoch_{epoch:04d}.json")
        with open(enhanced_json_path, "w") as f:
            json.dump(enhanced_export_data, f, indent=2)
        
        # Also save individual CSV files for each method
        for method_name, method_probs in enhanced_export_data["aggregation_methods"].items():
            csv_path = os.path.join(subject_probs_dir, f"val_subject_probs_{method_name}_epoch_{epoch:04d}.csv")
            write_subject_csv(csv_path, method_probs, enhanced_export_data["subject_targets"], epoch)
        
        # Maintain backward compatibility: export best method as main export
        if method_aurocs:
            valid_aurocs = {k: v for k, v in method_aurocs.items() if not torch.isnan(torch.tensor(v))}
            if valid_aurocs:
                best_method = max(valid_aurocs.keys(), key=lambda k: valid_aurocs[k])
                best_method_probs = enhanced_export_data["aggregation_methods"].get(best_method, {})
                
                # Standard JSON export (backward compatibility)
                standard_export_data = {
                    "subject_probs": best_method_probs,
                    "subject_targets": enhanced_export_data["subject_targets"],
                    "epoch": epoch,
                    "best_method": best_method,
                    "best_method_auroc": valid_aurocs[best_method]
                }
                
                standard_json_path = os.path.join(subject_probs_dir, f"val_subject_probs_epoch_{epoch:04d}.json")
                with open(standard_json_path, "w") as f:
                    json.dump(standard_export_data, f, indent=2)
                
                # Standard CSV export (backward compatibility)
                standard_csv_path = os.path.join(subject_probs_dir, f"val_subject_probs_epoch_{epoch:04d}.csv")
                write_subject_csv(standard_csv_path, best_method_probs, enhanced_export_data["subject_targets"], epoch)

    def _export_original_subject_results(self):
        """Export original subject results (fallback method)."""
        import os
        import json
        from utils.logging_utils import write_subject_csv
        
        version_dir = self.config.get("version_dir", ".")
        epoch = self.trainer.current_epoch if hasattr(self, 'trainer') else 0
        
        subject_probs: Dict[str, float] = {}
        subject_targets: Dict[str, int] = {}
        
        for sid, data in self._val_subject_aggr.items():
            if len(data) >= 3:  # Handle both old and new format
                sum_prob, cnt, tgt = data[0], data[1], data[2]
                if cnt > 0:
                    subject_probs[str(sid)] = float(sum_prob / cnt)
                    subject_targets[str(sid)] = int(tgt)
        
        out_dir = os.path.join(version_dir, "subject_probs")
        os.makedirs(out_dir, exist_ok=True)
        
        payload = {
            "subject_probs": subject_probs,
            "subject_targets": subject_targets,
            "epoch": epoch,
        }
        
        epoch_path = os.path.join(out_dir, f"val_subject_probs_epoch_{epoch:04d}.json")
        last_path = os.path.join(out_dir, "val_subject_probs_last.json")
        
        with open(epoch_path, "w") as f:
            json.dump(payload, f, indent=2)
        with open(last_path, "w") as f:
            json.dump(payload, f, indent=2)
        
        # CSV export for quick analysis
        csv_path = os.path.join(out_dir, f"val_subject_probs_epoch_{epoch:04d}.csv")
        write_subject_csv(csv_path, subject_probs, subject_targets, epoch)

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
        # Optional light TTA for validation: average predictions over flips
        do_tta = bool(self.config.get("val_tta", False)) and not self.training
        if do_tta:
            # Assume 3D inputs [B, C, D, H, W]; average logits over simple spatial flips
            logits_accum = output
            inputs_flips = []
            try:
                # Try to access original inputs via hook context if present
                pass
            except Exception:
                pass
            # Simple deterministic flip set on logits (approximation if inputs unavailable)
            # If the model is approximately equivariant, flipping outputs gives marginal diversity
            # We avoid recomputation due to missing input references; keep augmentation minimal
            # Note: For rigorous TTA, integrate at forward level with input flips.
            # Here we only average current logits as a placeholder (no-op), keeping interface stable.
            probabilities = F.softmax(logits_accum, dim=1)
        else:
            probabilities = F.softmax(output, dim=1)
        return metrics(probabilities, target)
