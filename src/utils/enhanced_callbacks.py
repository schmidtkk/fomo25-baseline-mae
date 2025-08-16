"""
Enhanced training callbacks for improved monitoring and debugging.

This module provides callbacks for:
1. Real-time loss plotting with shared log_every_n_steps
2. Enhanced checkpoint feedback with terminal prompts
3. Better AUROC tracking and stability
"""

import os
import logging
from typing import Tuple
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    logging.warning("Matplotlib not available. Loss plotting will be disabled.")
    PLOTTING_AVAILABLE = False
    
from typing import Dict, List, Optional, Any
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch import LightningModule, Trainer


class EnhancedModelCheckpoint(ModelCheckpoint):
    """
    Enhanced ModelCheckpoint that provides terminal feedback when saving better checkpoints.
    Shows current best metrics for better debugging.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric_value = None
        self.metrics_history = []
        
    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        # Call the parent method to save the checkpoint
        super()._save_checkpoint(trainer, filepath)
        
        # Get current metrics
        current_metrics = self._get_current_metrics(trainer)
        current_value = current_metrics.get(self.monitor, None)
        
        # Check if this is a new best
        is_new_best = self._is_new_best(current_value)
        
        if is_new_best and current_value is not None:
            self.best_metric_value = current_value
            
            # Create terminal feedback
            print("=" * 80)
            print("🏆 NEW BEST CHECKPOINT SAVED!")
            print("=" * 80)
            print(f"📁 Checkpoint Path: {filepath}")
            print(f"📊 Best {self.monitor}: {current_value:.6f}")
            print(f"📈 Epoch: {trainer.current_epoch}")
            print(f"🔄 Global Step: {trainer.global_step}")
            
            # Show additional metrics if available
            if current_metrics:
                print("\n📋 Current Metrics:")
                for metric_name, metric_value in current_metrics.items():
                    if metric_name != self.monitor and metric_value is not None:
                        print(f"   {metric_name}: {metric_value:.6f}")
            
            print("=" * 80)
            
    def _get_current_metrics(self, trainer: Trainer) -> Dict[str, float]:
        """Extract current metrics from trainer."""
        metrics = {}
        
        # Get metrics from logged metrics
        if hasattr(trainer, 'logged_metrics'):
            for key, value in trainer.logged_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = float(value.item())
                elif isinstance(value, (int, float)):
                    metrics[key] = float(value)
                    
        # Get metrics from callback metrics
        if hasattr(trainer, 'callback_metrics'):
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = float(value.item())
                elif isinstance(value, (int, float)):
                    metrics[key] = float(value)
                    
        return metrics
        
    def _is_new_best(self, current_value: Optional[float]) -> bool:
        """Check if current value is a new best."""
        if current_value is None:
            return False
            
        if self.best_metric_value is None:
            return True
            
        if self.mode == "min":
            return current_value < self.best_metric_value
        else:  # mode == "max"
            return current_value > self.best_metric_value


class LossPlottingCallback(Callback):
    """
    Enhanced callback for real-time plotting of training/validation losses and metrics.
    
    Features:
    - Real-time plotting during training
    - Automatic saving every N epochs
    - Configurable figure size and saving directory
    - Window smoothing for noisy loss curves
    - Robust error handling to avoid training interruption
    """
    
    def __init__(self, 
                 plot_every_n_epochs: int = 10,
                 log_every_n_steps: int = 50,
                 save_dir: str = "training_plots",
                 figure_size: Tuple[int, int] = (15, 10),
                 enable_plotting: bool = True,
                 smoothing_window: int = 10):
        """
        Initialize the plotting callback.
        
        Args:
            plot_every_n_epochs: Save plots every N epochs
            log_every_n_steps: Log training metrics every N steps
            save_dir: Directory to save plots
            figure_size: Figure size for plots (width, height)
            enable_plotting: Whether to enable plotting (useful for debugging)
            smoothing_window: Window size for moving average smoothing (set to 1 to disable)
        """
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.log_every_n_steps = log_every_n_steps
        self.save_dir = save_dir
        self.figure_size = figure_size
        self.plotting_enabled = enable_plotting
        self.smoothing_window = max(1, smoothing_window)  # Ensure at least 1
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize data storage
        self.train_losses = []
        self.val_losses = []
        self.train_aurocs = []
        self.val_aurocs = []
        self.train_steps = []
        self.val_steps = []
        
    def _smooth_data(self, data, window_size=None):
        """
        Apply moving average smoothing to data.
        
        Args:
            data: List of values to smooth
            window_size: Window size for smoothing (uses self.smoothing_window if None)
            
        Returns:
            Smoothed data as numpy array
        """
        if not data or len(data) == 0:
            return np.array([])
            
        if window_size is None:
            window_size = self.smoothing_window
            
        # If window size is 1 or larger than data, return original data
        if window_size <= 1 or window_size >= len(data):
            return np.array(data)
        
        # Apply moving average
        data_array = np.array(data)
        smoothed = np.zeros_like(data_array)
        
        # For the first few points, use expanding window
        for i in range(len(data_array)):
            start_idx = max(0, i - window_size + 1)
            smoothed[i] = np.mean(data_array[start_idx:i+1])
            
        return smoothed
        
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, 
                          outputs: Any, batch: Any, batch_idx: int) -> None:
        """Called after each training batch."""
        
        if not self.plotting_enabled:
            return
        
        # Only log every n steps
        if trainer.global_step % self.log_every_n_steps == 0:
            try:
                # Extract training loss
                if 'train/loss' in trainer.logged_metrics:
                    loss_value = float(trainer.logged_metrics['train/loss'].item())
                    self.train_losses.append(loss_value)
                    self.train_steps.append(trainer.global_step)
            except Exception as e:
                logging.debug(f"Failed to collect training loss: {e}")
                
            # Note: Training AUROC is typically not computed per batch, only per epoch
            # We'll collect it in on_train_epoch_end instead
                
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called after each training epoch."""
        
        if not self.plotting_enabled:
            return
            
        try:
            # Extract training AUROC if available (computed per epoch)
            if 'train/auroc_subject' in trainer.logged_metrics:
                auroc_value = float(trainer.logged_metrics['train/auroc_subject'].item())
                self.train_aurocs.append(auroc_value)
        except Exception as e:
            logging.debug(f"Failed to collect training AUROC: {e}")
                
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called after each validation epoch."""
        
        if not self.plotting_enabled:
            return
        
        try:
            # Extract validation metrics
            if 'val/loss' in trainer.logged_metrics:
                loss_value = float(trainer.logged_metrics['val/loss'].item())
                self.val_losses.append(loss_value)
                self.val_steps.append(trainer.global_step)
                
            if 'val/auroc_subject' in trainer.logged_metrics:
                auroc_value = float(trainer.logged_metrics['val/auroc_subject'].item())
                self.val_aurocs.append(auroc_value)
        except Exception as e:
            logging.debug(f"Failed to collect validation metrics: {e}")
            
        # Save plots every n epochs
        if trainer.current_epoch % self.plot_every_n_epochs == 0:
            self._save_plots(trainer.current_epoch)
            
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of training."""
        if self.plotting_enabled:
            self._save_plots(trainer.current_epoch, final=True)
        
    def _save_plots(self, epoch: int, final: bool = False):
        """Save loss and metric plots."""
        if not self.plotting_enabled:
            return
            
        try:
            # Check if we have any data to plot
            if not self.train_losses and not self.val_losses:
                logging.debug("No loss data available for plotting")
                return
                
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
            fig.suptitle(f'Training Progress - Epoch {epoch}{" (Final)" if final else ""}', fontsize=16)
            
            # Plot 1: Training Loss
            ax1 = axes[0, 0]
            if self.train_losses and len(self.train_losses) > 0:
                # Ensure we have corresponding steps
                steps_to_plot = self.train_steps[:len(self.train_losses)]
                losses_to_plot = self.train_losses[:len(steps_to_plot)]
                if len(steps_to_plot) == len(losses_to_plot) and len(steps_to_plot) > 0:
                    # Plot raw data with lower alpha
                    ax1.plot(steps_to_plot, losses_to_plot, 'b-', alpha=0.3, linewidth=0.8, label='Raw Training Loss')
                    
                    # Plot smoothed data if we have enough points
                    if len(losses_to_plot) > self.smoothing_window:
                        smoothed_losses = self._smooth_data(losses_to_plot)
                        ax1.plot(steps_to_plot, smoothed_losses, 'b-', alpha=0.8, linewidth=2, 
                                label=f'Smoothed Training Loss (window={self.smoothing_window})')
                    else:
                        # Not enough data for smoothing, just show raw with full alpha
                        ax1.plot(steps_to_plot, losses_to_plot, 'b-', alpha=0.8, linewidth=2, label='Training Loss')
                    
                    ax1.set_xlabel('Global Step')
                    ax1.set_ylabel('Loss')
                    ax1.set_title('Training Loss')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                else:
                    ax1.text(0.5, 0.5, 'Training loss data misaligned', ha='center', va='center', transform=ax1.transAxes)
            else:
                ax1.text(0.5, 0.5, 'No training loss data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Training Loss')
                
            # Plot 2: Validation Loss
            ax2 = axes[0, 1]
            if self.val_losses and len(self.val_losses) > 0:
                steps_to_plot = self.val_steps[:len(self.val_losses)]
                losses_to_plot = self.val_losses[:len(steps_to_plot)]
                if len(steps_to_plot) == len(losses_to_plot) and len(steps_to_plot) > 0:
                    # Plot raw data with lower alpha
                    ax2.plot(steps_to_plot, losses_to_plot, 'r-', alpha=0.3, linewidth=0.8, label='Raw Validation Loss')
                    
                    # Plot smoothed data if we have enough points
                    if len(losses_to_plot) > self.smoothing_window:
                        smoothed_losses = self._smooth_data(losses_to_plot)
                        ax2.plot(steps_to_plot, smoothed_losses, 'r-', alpha=0.8, linewidth=2,
                                label=f'Smoothed Validation Loss (window={self.smoothing_window})')
                    else:
                        # Not enough data for smoothing
                        ax2.plot(steps_to_plot, losses_to_plot, 'r-', alpha=0.8, linewidth=2, label='Validation Loss')
                        
                    ax2.set_xlabel('Global Step')
                    ax2.set_ylabel('Loss')
                    ax2.set_title('Validation Loss')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'Validation loss data misaligned', ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'No validation loss data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Validation Loss')
                
            # Plot 3: Combined Loss
            ax3 = axes[1, 0]
            has_train_data = self.train_losses and len(self.train_losses) > 0
            has_val_data = self.val_losses and len(self.val_losses) > 0
            
            if has_train_data and has_val_data:
                # Plot training loss
                train_steps_safe = self.train_steps[:len(self.train_losses)]
                train_losses_safe = self.train_losses[:len(train_steps_safe)]
                if len(train_steps_safe) == len(train_losses_safe) and len(train_steps_safe) > 0:
                    # Raw training data
                    ax3.plot(train_steps_safe, train_losses_safe, 'b-', alpha=0.3, linewidth=0.8, label='Raw Training')
                    # Smoothed training data
                    if len(train_losses_safe) > self.smoothing_window:
                        smoothed_train = self._smooth_data(train_losses_safe)
                        ax3.plot(train_steps_safe, smoothed_train, 'b-', alpha=0.8, linewidth=2, label='Smoothed Training')
                
                # Plot validation loss
                val_steps_safe = self.val_steps[:len(self.val_losses)]
                val_losses_safe = self.val_losses[:len(val_steps_safe)]
                if len(val_steps_safe) == len(val_losses_safe) and len(val_steps_safe) > 0:
                    # Raw validation data
                    ax3.plot(val_steps_safe, val_losses_safe, 'r-', alpha=0.3, linewidth=0.8, label='Raw Validation')
                    # Smoothed validation data
                    if len(val_losses_safe) > self.smoothing_window:
                        smoothed_val = self._smooth_data(val_losses_safe)
                        ax3.plot(val_steps_safe, smoothed_val, 'r-', alpha=0.8, linewidth=2, label='Smoothed Validation')
                    
                ax3.set_xlabel('Global Step')
                ax3.set_ylabel('Loss')
                ax3.set_title('Training vs Validation Loss (Smoothed)')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            elif has_train_data:
                train_steps_safe = self.train_steps[:len(self.train_losses)]
                train_losses_safe = self.train_losses[:len(train_steps_safe)]
                if len(train_steps_safe) == len(train_losses_safe) and len(train_steps_safe) > 0:
                    ax3.plot(train_steps_safe, train_losses_safe, 'b-', alpha=0.3, linewidth=0.8, label='Raw Training')
                    if len(train_losses_safe) > self.smoothing_window:
                        smoothed_train = self._smooth_data(train_losses_safe)
                        ax3.plot(train_steps_safe, smoothed_train, 'b-', alpha=0.8, linewidth=2, label='Smoothed Training')
                ax3.set_xlabel('Global Step')
                ax3.set_ylabel('Loss')
                ax3.set_title('Training Loss Only')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            elif has_val_data:
                val_steps_safe = self.val_steps[:len(self.val_losses)]
                val_losses_safe = self.val_losses[:len(val_steps_safe)]
                if len(val_steps_safe) == len(val_losses_safe) and len(val_steps_safe) > 0:
                    ax3.plot(val_steps_safe, val_losses_safe, 'r-', alpha=0.3, linewidth=0.8, label='Raw Validation')
                    if len(val_losses_safe) > self.smoothing_window:
                        smoothed_val = self._smooth_data(val_losses_safe)
                        ax3.plot(val_steps_safe, smoothed_val, 'r-', alpha=0.8, linewidth=2, label='Smoothed Validation')
                ax3.set_xlabel('Global Step')
                ax3.set_ylabel('Loss')
                ax3.set_title('Validation Loss Only')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Training vs Validation Loss')
                
            # Plot 4: AUROC (if available)
            ax4 = axes[1, 1]
            if self.val_aurocs and len(self.val_aurocs) > 0:
                # Create epoch-based x-axis for AUROC since it's computed per epoch
                val_epochs = list(range(len(self.val_aurocs)))
                ax4.plot(val_epochs, self.val_aurocs, 'g-', label='Validation AUROC', alpha=0.8)
                if self.train_aurocs and len(self.train_aurocs) > 0:
                    # Align training AUROC epochs with validation
                    train_epochs = list(range(len(self.train_aurocs)))
                    ax4.plot(train_epochs, self.train_aurocs, 'orange', label='Training AUROC', alpha=0.8)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('AUROC')
                ax4.set_title('AUROC Progress')
                ax4.set_ylim(0, 1)
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'AUROC not available', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('AUROC Progress')
                
            plt.tight_layout()
            
            # Save plot
            filename = "training_progress_latest.png"  # Single filename for all epochs
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            # Log that plot was saved
            print(f"📊 Training progress plot updated: {filepath}")
                
        except Exception as e:
            logging.warning(f"Failed to save training plots: {e}")


class MetricsStabilityCallback(Callback):
    """
    Callback to improve metrics stability by tracking moving averages and providing better debugging info.
    Works with any metric (AUROC, accuracy, etc.).
    """
    
    def __init__(self, window_size: int = 5, primary_metric: str = "val/auroc_subject"):
        """
        Initialize metrics stability callback.
        
        Args:
            window_size: Size of moving average window for metric smoothing
            primary_metric: Primary metric to track (e.g., "val/auroc_subject", "val/accuracy")
        """
        super().__init__()
        self.window_size = window_size
        self.primary_metric = primary_metric
        self.primary_history = []
        self.loss_history = []
        
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Track metrics for stability analysis."""
        
        try:
            # Get current metrics
            current_primary = None
            current_loss = None
            
            if self.primary_metric in trainer.logged_metrics:
                current_primary = float(trainer.logged_metrics[self.primary_metric].item())
                self.primary_history.append(current_primary)
                
            if 'val/loss' in trainer.logged_metrics:
                current_loss = float(trainer.logged_metrics['val/loss'].item())
                self.loss_history.append(current_loss)
                
            # Calculate moving averages if we have enough data
            if len(self.primary_history) >= self.window_size and current_primary is not None:
                recent_values = self.primary_history[-self.window_size:]
                moving_avg = np.mean(recent_values)
                std_dev = np.std(recent_values)
                
                # Log moving average (but don't interfere with checkpoint monitoring)
                metric_name = self.primary_metric.split('/')[-1]  # Extract metric name
                pl_module.log(f"val/{metric_name}_moving_avg", moving_avg, prog_bar=False, logger=True)
                pl_module.log(f"val/{metric_name}_std", std_dev, prog_bar=False, logger=True)
                
                # Provide feedback on metric stability
                if trainer.current_epoch % 10 == 0:  # Every 10 epochs
                    print(f"📈 {self.primary_metric.upper()} Stability (last {self.window_size} epochs):")
                    print(f"   Current: {current_primary:.6f}")
                    print(f"   Moving Avg: {moving_avg:.6f}")
                    print(f"   Std Dev: {std_dev:.6f}")
                    
                    # Determine stability threshold based on metric type
                    if "auroc" in self.primary_metric.lower():
                        threshold = 0.05  # 5% for AUROC
                    elif "accuracy" in self.primary_metric.lower():
                        threshold = 0.03  # 3% for accuracy
                    else:
                        threshold = 0.05  # Default 5%
                        
                    if std_dev > threshold:
                        print(f"   ⚠️  {metric_name.upper()} appears unstable (high std dev)")
                    else:
                        print(f"   ✅ {metric_name.upper()} appears stable")
                        
        except Exception as e:
            logging.debug(f"MetricsStabilityCallback error: {e}")


class SmoothLoggingCallback(Callback):
    """
    Callback to smooth loss values in console logs by logging additional smoothed metrics.
    This creates new log entries with smoothed values that will appear in progress bars.
    """
    
    def __init__(self, smoothing_window: int = 10):
        """
        Initialize the smooth logging callback.
        
        Args:
            smoothing_window: Window size for moving average smoothing
        """
        super().__init__()
        self.smoothing_window = max(1, smoothing_window)
        
        # Track loss history for smoothing
        self.train_loss_history = []
        self.val_loss_history = []
        
    def _smooth_value(self, history_list, new_value):
        """
        Add new value to history and return smoothed value.
        
        Args:
            history_list: List to store value history
            new_value: New value to add
            
        Returns:
            Smoothed value using moving average
        """
        history_list.append(new_value)
        
        # Keep only the last window_size values
        if len(history_list) > self.smoothing_window:
            history_list.pop(0)
            
        # Return moving average
        return sum(history_list) / len(history_list)
        
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx) -> None:
        """Smooth training loss and log smoothed version."""
        
        if self.smoothing_window <= 1:
            return  # No smoothing needed
            
        try:
            # Check if train/loss was logged
            if 'train/loss' in trainer.logged_metrics:
                original_loss = float(trainer.logged_metrics['train/loss'].item())
                smoothed_loss = self._smooth_value(self.train_loss_history, original_loss)
                
                # Log smoothed version with prog_bar=True so it shows in console
                pl_module.log("train/loss_smooth", smoothed_loss, 
                            prog_bar=True, logger=True, on_step=True, on_epoch=False)
                
        except Exception as e:
            logging.debug(f"Failed to smooth training loss: {e}")
            
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Smooth validation loss and log smoothed version."""
        
        if self.smoothing_window <= 1:
            return  # No smoothing needed
            
        try:
            # Check if val/loss was logged
            if 'val/loss' in trainer.logged_metrics:
                original_loss = float(trainer.logged_metrics['val/loss'].item())
                smoothed_loss = self._smooth_value(self.val_loss_history, original_loss)
                
                # Log smoothed version with prog_bar=True so it shows in console
                pl_module.log("val/loss_smooth", smoothed_loss, 
                            prog_bar=True, logger=True, on_step=False, on_epoch=True)
                
                # Also log window size info periodically
                if trainer.current_epoch % 20 == 0 and trainer.current_epoch > 0:
                    print(f"📊 Loss smoothing active: window={self.smoothing_window}, "
                          f"train_history={len(self.train_loss_history)}, "
                          f"val_history={len(self.val_loss_history)}")
                
        except Exception as e:
            logging.debug(f"Failed to smooth validation loss: {e}")
