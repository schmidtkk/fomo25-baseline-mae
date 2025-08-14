"""
Enhanced early stopping with ensemble metrics and smoothing for stable training.
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from collections import defaultdict
import warnings
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class RobustEarlyStopping(EarlyStopping):
    """
    Enhanced early stopping that uses ensemble of metrics and temporal smoothing
    to handle noisy validation metrics in small datasets.
    """
    
    def __init__(self, 
                 primary_metric: str = "val/auroc_subject",
                 ensemble_metrics: Optional[List[str]] = None,
                 ensemble_weight: float = 0.7,
                 smoothing_window: int = 3,
                 min_epochs_before_stop: int = 15,
                 **kwargs):
        """
        Initialize robust early stopping.
        
        Args:
            primary_metric: Primary metric to monitor (backward compatibility)
            ensemble_metrics: List of metrics to ensemble (e.g., different aggregation AUROCs)
            ensemble_weight: Weight for ensemble metric vs primary metric
            smoothing_window: Number of epochs to smooth over
            min_epochs_before_stop: Minimum epochs before early stopping can trigger
            **kwargs: Additional arguments for base EarlyStopping
        """
        # Initialize base EarlyStopping; avoid passing duplicate 'monitor'
        if 'monitor' in kwargs:
            super().__init__(**kwargs)
            self.primary_metric = kwargs['monitor']
        else:
            super().__init__(monitor=primary_metric, **kwargs)
        
        self.primary_metric = getattr(self, 'primary_metric', primary_metric)
        self.ensemble_metrics = ensemble_metrics or [
            "val/auroc_subject_mean_logit",
            "val/auroc_subject_noisy_or",
            "val/auroc_subject_top_k_3"
        ]
        self.ensemble_weight = ensemble_weight
        self.smoothing_window = smoothing_window
        self.min_epochs_before_stop = min_epochs_before_stop
        
        # Tracking
        self.metric_history = defaultdict(list)
        self.ensemble_history = []
        self.smoothed_history = []
        
        # Keep EarlyStopping.monitor as the primary metric so Lightning doesn't error.
        # We still compute an internal ensemble metric for our own stopping logic.
        
        print(f"RobustEarlyStopping initialized:")
        print(f"  Primary metric: {self.primary_metric}")
        print(f"  Ensemble metrics: {self.ensemble_metrics}")
        print(f"  Ensemble weight: {self.ensemble_weight}")
        print(f"  Smoothing window: {self.smoothing_window}")
        print(f"  Min epochs before stop: {self.min_epochs_before_stop}")
    
    def _get_ensemble_metric(self, trainer) -> float:
        """
        Compute ensemble metric from multiple aggregation AUROCs.
        
        Returns:
            Ensemble metric value
        """
        current_metrics = {}
        
        # Get logged metrics from trainer
        if hasattr(trainer, 'logged_metrics'):
            logged_metrics = trainer.logged_metrics
        elif hasattr(trainer, 'callback_metrics'):
            logged_metrics = trainer.callback_metrics
        else:
            return float('-inf')
        
        # Extract primary metric
        primary_value = None
        if self.primary_metric in logged_metrics:
            try:
                primary_value = float(logged_metrics[self.primary_metric])
            except:
                primary_value = None
        
        # Extract ensemble metrics
        ensemble_values = []
        for metric in self.ensemble_metrics:
            if metric in logged_metrics:
                try:
                    value = float(logged_metrics[metric])
                    if not np.isnan(value):
                        ensemble_values.append(value)
                except:
                    continue
        
        # Compute ensemble score
        if ensemble_values:
            ensemble_mean = np.mean(ensemble_values)
        else:
            ensemble_mean = None
        
        # Combine primary and ensemble
        if primary_value is not None and ensemble_mean is not None:
            # Weighted combination
            combined_metric = (self.ensemble_weight * ensemble_mean + 
                             (1 - self.ensemble_weight) * primary_value)
        elif primary_value is not None:
            # Fallback to primary only
            combined_metric = primary_value
        elif ensemble_mean is not None:
            # Fallback to ensemble only
            combined_metric = ensemble_mean
        else:
            # No metrics available
            return float('-inf')
        
        return combined_metric
    
    def _apply_smoothing(self, values: List[float]) -> float:
        """
        Apply temporal smoothing to reduce noise.
        
        Args:
            values: List of recent metric values
            
        Returns:
            Smoothed metric value
        """
        if not values:
            return float('-inf')
        
        if len(values) < self.smoothing_window:
            # Not enough data for smoothing
            return values[-1]
        
        # Use recent values for smoothing
        recent_values = values[-self.smoothing_window:]
        
        # Simple moving average
        smoothed = np.mean(recent_values)
        
        return smoothed
    
    def _check_early_stopping(self, trainer) -> bool:
        """
        Check if early stopping should trigger.
        
        Returns:
            True if training should stop
        """
        current_epoch = trainer.current_epoch
        
        # Don't stop before minimum epochs
        if current_epoch < self.min_epochs_before_stop:
            return False
        
        # Get current ensemble metric
        current_metric = self._get_ensemble_metric(trainer)
        
        if current_metric == float('-inf'):
            # No valid metrics available
            return False
        
        # Store in history
        self.ensemble_history.append(current_metric)
        
        # Apply smoothing
        smoothed_metric = self._apply_smoothing(self.ensemble_history)
        self.smoothed_history.append(smoothed_metric)
        
        # Check improvement using smoothed values
        if len(self.smoothed_history) < 2:
            return False
        
        # Find best smoothed value so far
        if self.mode == "max":
            best_smoothed = max(self.smoothed_history[:-1])  # Exclude current
            improvement = smoothed_metric - best_smoothed
        else:
            best_smoothed = min(self.smoothed_history[:-1])  # Exclude current
            improvement = best_smoothed - smoothed_metric
        
        # Check if improvement is significant
        if improvement > self.min_delta:
            # Improvement detected, reset patience
            self.wait_count = 0
            return False
        else:
            # No improvement
            self.wait_count += 1
            
            if self.wait_count >= self.patience:
                return True
            
            return False
    
    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        # Store detailed metrics for analysis
        if hasattr(trainer, 'logged_metrics'):
            current_epoch = trainer.current_epoch
            for metric_name, value in trainer.logged_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.metric_history[metric_name].append(value)
        
        # Check early stopping
        should_stop = self._check_early_stopping(trainer)
        
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True
            
            # Log stopping information
            current_metric = self._get_ensemble_metric(trainer)
            print(f"\nEarly stopping triggered at epoch {self.stopped_epoch}")
            print(f"Current ensemble metric: {current_metric:.4f}")
            print(f"Best smoothed metric: {max(self.smoothed_history) if self.mode == 'max' else min(self.smoothed_history):.4f}")
            print(f"Patience exhausted: {self.wait_count}/{self.patience}")
    
    def get_stopping_summary(self) -> Dict[str, Any]:
        """
        Get summary of early stopping behavior for analysis.
        
        Returns:
            Dictionary with stopping summary
        """
        if not self.ensemble_history:
            return {}
        
        summary = {
            'stopped_epoch': getattr(self, 'stopped_epoch', None),
            'total_epochs': len(self.ensemble_history),
            'final_ensemble_metric': self.ensemble_history[-1] if self.ensemble_history else None,
            'best_ensemble_metric': max(self.ensemble_history) if self.mode == 'max' else min(self.ensemble_history),
            'final_smoothed_metric': self.smoothed_history[-1] if self.smoothed_history else None,
            'best_smoothed_metric': max(self.smoothed_history) if self.mode == 'max' else min(self.smoothed_history),
            'patience_used': self.wait_count,
            'patience_limit': self.patience,
            'metric_history_length': len(self.ensemble_history)
        }
        
        return summary
    
    def save_stopping_analysis(self, save_path: str):
        """Save detailed stopping analysis to file."""
        try:
            import json
            import os
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            analysis = {
                'stopping_summary': self.get_stopping_summary(),
                'ensemble_history': self.ensemble_history,
                'smoothed_history': self.smoothed_history,
                'metric_history': dict(self.metric_history),
                'config': {
                    'primary_metric': self.primary_metric,
                    'ensemble_metrics': self.ensemble_metrics,
                    'ensemble_weight': self.ensemble_weight,
                    'smoothing_window': self.smoothing_window,
                    'patience': self.patience,
                    'min_delta': self.min_delta,
                    'mode': self.mode
                }
            }
            
            with open(save_path, 'w') as f:
                json.dump(analysis, f, indent=2)
                
            print(f"Early stopping analysis saved to {save_path}")
            
        except Exception as e:
            warnings.warn(f"Error saving early stopping analysis: {e}")


class AdaptiveEarlyStopping(RobustEarlyStopping):
    """
    Adaptive early stopping that adjusts patience based on training dynamics.
    """
    
    def __init__(self, 
                 initial_patience: int = 12,
                 max_patience: int = 25,
                 patience_factor: float = 1.5,
                 adaptation_window: int = 10,
                 **kwargs):
        """
        Initialize adaptive early stopping.
        
        Args:
            initial_patience: Starting patience value
            max_patience: Maximum patience allowed
            patience_factor: Factor to increase patience by
            adaptation_window: Window to assess improvement trend
            **kwargs: Additional arguments for RobustEarlyStopping
        """
        super().__init__(patience=initial_patience, **kwargs)
        
        self.initial_patience = initial_patience
        self.max_patience = max_patience
        self.patience_factor = patience_factor
        self.adaptation_window = adaptation_window
        self.patience_adjustments = []
        
    def _assess_improvement_trend(self) -> str:
        """
        Assess whether improvement trend is positive, negative, or stable.
        
        Returns:
            Trend assessment: 'improving', 'declining', 'stable'
        """
        if len(self.smoothed_history) < self.adaptation_window:
            return 'stable'
        
        recent_values = self.smoothed_history[-self.adaptation_window:]
        
        # Calculate trend using linear regression slope
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        # Determine trend based on slope and mode
        if self.mode == 'max':
            if slope > self.min_delta:
                return 'improving'
            elif slope < -self.min_delta:
                return 'declining'
            else:
                return 'stable'
        else:  # mode == 'min'
            if slope < -self.min_delta:
                return 'improving'
            elif slope > self.min_delta:
                return 'declining'
            else:
                return 'stable'
    
    def _adjust_patience(self, trainer):
        """Adjust patience based on improvement trend."""
        current_epoch = trainer.current_epoch
        
        # Only adjust patience periodically
        if current_epoch % self.adaptation_window != 0:
            return
        
        if len(self.smoothed_history) < self.adaptation_window:
            return
        
        trend = self._assess_improvement_trend()
        old_patience = self.patience
        
        if trend == 'improving' and self.patience < self.max_patience:
            # Increase patience if we're still improving
            new_patience = min(int(self.patience * self.patience_factor), self.max_patience)
            self.patience = new_patience
            
            adjustment = {
                'epoch': current_epoch,
                'trend': trend,
                'old_patience': old_patience,
                'new_patience': new_patience,
                'reason': 'increasing_due_to_improvement'
            }
            self.patience_adjustments.append(adjustment)
            
            print(f"Patience increased from {old_patience} to {new_patience} due to improving trend")
        
        elif trend == 'declining' and self.wait_count > self.patience // 2:
            # Decrease patience if clearly declining
            new_patience = max(self.initial_patience, int(self.patience / self.patience_factor))
            self.patience = new_patience
            
            adjustment = {
                'epoch': current_epoch,
                'trend': trend,
                'old_patience': old_patience,
                'new_patience': new_patience,
                'reason': 'decreasing_due_to_decline'
            }
            self.patience_adjustments.append(adjustment)
            
            print(f"Patience decreased from {old_patience} to {new_patience} due to declining trend")
    
    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        # First, do the standard robust early stopping check
        super().on_validation_end(trainer, pl_module)
        
        # Then, adjust patience if needed
        if not trainer.should_stop:  # Only adjust if we're not already stopping
            self._adjust_patience(trainer)


def create_enhanced_early_stopping(stopping_type: str = "robust",
                                  primary_metric: str = "val/auroc_subject",
                                  **kwargs) -> EarlyStopping:
    """
    Factory function to create enhanced early stopping callbacks.
    
    Args:
        stopping_type: Type of early stopping ('robust' or 'adaptive')
        primary_metric: Primary metric to monitor
        **kwargs: Additional arguments for the stopping callback
        
    Returns:
        Enhanced early stopping callback
    """
    if stopping_type == "robust":
        return RobustEarlyStopping(primary_metric=primary_metric, **kwargs)
    elif stopping_type == "adaptive":
        return AdaptiveEarlyStopping(primary_metric=primary_metric, **kwargs)
    else:
        raise ValueError(f"Unknown stopping type: {stopping_type}")
