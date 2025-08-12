"""
Real-time training visualization for monitoring training dynamics and subject-level aggregation methods.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mstyle
from collections import defaultdict
from typing import Dict, List, Optional, Any
import seaborn as sns
from pathlib import Path

# Set matplotlib backend for server environments
plt.switch_backend('Agg')
mstyle.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    """
    Real-time training visualization system that creates comprehensive dashboards
    during training to monitor progress and compare aggregation methods.
    """
    
    def __init__(self, save_dir: str, enable_live_plots: bool = True, 
                 update_frequency: int = 1, figsize: tuple = (20, 15)):
        """
        Initialize the training visualizer.
        
        Args:
            save_dir: Directory to save plots and metrics
            enable_live_plots: Whether to generate plots during training
            update_frequency: Update plots every N epochs
            figsize: Figure size for the dashboard
        """
        self.save_dir = Path(save_dir)
        self.plots_dir = self.save_dir / "training_plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_live = enable_live_plots
        self.update_frequency = update_frequency
        self.figsize = figsize
        
        # Metrics tracking
        self.metrics_history = defaultdict(list)
        self.epoch_history = []
        self.subject_results_history = []
        
        # Plot styling
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = sns.color_palette("husl", 10)
        
    def update_metrics(self, epoch: int, metrics_dict: Dict[str, float], 
                      subject_results: Optional[Dict] = None):
        """
        Update metrics history and generate plots if enabled.
        
        Args:
            epoch: Current epoch number
            metrics_dict: Dictionary of logged metrics
            subject_results: Results from different aggregation methods
        """
        self.epoch_history.append(epoch)
        
        # Store metrics
        for metric_name, value in metrics_dict.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.metrics_history[metric_name].append(value)
        
        # Store subject results
        if subject_results:
            self.subject_results_history.append({
                'epoch': epoch,
                'results': subject_results
            })
        
        # Generate plots
        if self.enable_live and (epoch % self.update_frequency == 0):
            self.generate_training_dashboard(epoch)
            
    def generate_training_dashboard(self, epoch: int):
        """Generate comprehensive training dashboard."""
        try:
            fig = plt.figure(figsize=self.figsize)
            
            # Create subplot grid
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. Loss curves
            ax1 = fig.add_subplot(gs[0, 0:2])
            self._plot_loss_curves(ax1)
            
            # 2. AUROC comparison
            ax2 = fig.add_subplot(gs[0, 2:4])
            self._plot_auroc_comparison(ax2)
            
            # 3. Subject probability distributions
            ax3 = fig.add_subplot(gs[1, 0:2])
            self._plot_subject_distributions(ax3)
            
            # 4. Early stopping progress
            ax4 = fig.add_subplot(gs[1, 2:4])
            self._plot_early_stopping_progress(ax4)
            
            # 5. Aggregation method ranking
            ax5 = fig.add_subplot(gs[2, 0:2])
            self._plot_method_ranking(ax5)
            
            # 6. Training stability metrics
            ax6 = fig.add_subplot(gs[2, 2:4])
            self._plot_training_stability(ax6)
            
            plt.suptitle(f'Training Dashboard - Epoch {epoch}', fontsize=16, fontweight='bold')
            
            # Save plot
            plot_path = self.plots_dir / f"dashboard_epoch_{epoch:04d}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Also save latest as current
            latest_path = self.plots_dir / "latest_dashboard.png"
            plt.savefig(latest_path, dpi=150, bbox_inches='tight')
            
            print(f"Training dashboard saved to {plot_path}")
            
        except Exception as e:
            print(f"Error generating training dashboard: {e}")
            plt.close('all')
    
    def _plot_loss_curves(self, ax):
        """Plot training and validation loss curves."""
        train_loss = self.metrics_history.get('train/loss', [])
        val_loss = self.metrics_history.get('val/loss', [])
        
        if train_loss:
            epochs = self.epoch_history[:len(train_loss)]
            ax.plot(epochs, train_loss, label='Train Loss', color=self.colors[0], linewidth=2)
        
        if val_loss:
            epochs = self.epoch_history[:len(val_loss)]
            ax.plot(epochs, val_loss, label='Val Loss', color=self.colors[1], linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_auroc_comparison(self, ax):
        """Plot comparison of different AUROC aggregation methods."""
        auroc_methods = [key for key in self.metrics_history.keys() 
                        if 'auroc_subject' in key and 'val/' in key]
        
        if not auroc_methods:
            ax.text(0.5, 0.5, 'No AUROC metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('AUROC Method Comparison')
            return
        
        for i, method in enumerate(auroc_methods):
            values = self.metrics_history[method]
            if values:
                epochs = self.epoch_history[:len(values)]
                method_name = method.replace('val/auroc_subject_', '').replace('val/auroc_subject', 'current')
                ax.plot(epochs, values, label=method_name, 
                       color=self.colors[i % len(self.colors)], 
                       linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUROC')
        ax.set_title('AUROC Method Comparison')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_subject_distributions(self, ax):
        """Plot subject probability distributions for latest epoch."""
        if not self.subject_results_history:
            ax.text(0.5, 0.5, 'No subject results available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Subject Probability Distributions')
            return
        
        latest_results = self.subject_results_history[-1]['results']
        
        # Plot probability distributions for different methods
        methods = ['mean_prob', 'mean_logit', 'noisy_or']
        colors_subset = self.colors[:len(methods)]
        
        for i, method in enumerate(methods):
            if method in latest_results:
                probs = [latest_results[sid][method] for sid in latest_results 
                        if method in latest_results[sid]]
                if probs:
                    ax.hist(probs, bins=20, alpha=0.6, label=method, 
                           color=colors_subset[i], density=True)
        
        ax.set_xlabel('Probability')
        ax.set_ylabel('Density')
        ax.set_title('Subject Probability Distributions (Latest Epoch)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_early_stopping_progress(self, ax):
        """Plot early stopping progress and patience."""
        # This would need to be fed from the early stopping callback
        # For now, show best metric progress
        auroc_values = self.metrics_history.get('val/auroc_subject', [])
        
        if auroc_values:
            epochs = self.epoch_history[:len(auroc_values)]
            ax.plot(epochs, auroc_values, label='Current AUROC', 
                   color=self.colors[0], linewidth=2)
            
            # Show best value so far
            best_value = max(auroc_values)
            best_epoch = epochs[np.argmax(auroc_values)]
            ax.axhline(y=best_value, color=self.colors[1], linestyle='--', 
                      label=f'Best: {best_value:.3f} (Epoch {best_epoch})')
            ax.axvline(x=best_epoch, color=self.colors[1], linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUROC')
        ax.set_title('Early Stopping Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_method_ranking(self, ax):
        """Plot ranking of aggregation methods over time."""
        if not self.subject_results_history:
            ax.text(0.5, 0.5, 'No subject results available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Aggregation Method Ranking')
            return
        
        # Extract method performance over time
        methods = ['mean_prob', 'mean_logit', 'noisy_or', 'top_k_3', 'max_prob']
        method_scores = {method: [] for method in methods}
        epochs = []
        
        for result in self.subject_results_history:
            epochs.append(result['epoch'])
            results = result['results']
            
            # Calculate AUROC for each method
            targets = [results[sid]['target'] for sid in results if 'target' in results[sid]]
            
            for method in methods:
                if method in results[list(results.keys())[0]]:  # Check if method exists
                    probs = [results[sid][method] for sid in results if method in results[sid]]
                    if len(set(targets)) >= 2 and len(probs) == len(targets):
                        # Simple AUROC calculation
                        auroc = self._calculate_simple_auroc(probs, targets)
                        method_scores[method].append(auroc)
                    else:
                        method_scores[method].append(np.nan)
                else:
                    method_scores[method].append(np.nan)
        
        # Plot method performance
        for i, method in enumerate(methods):
            scores = method_scores[method]
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                ax.plot(epochs[:len(scores)], scores, label=method, 
                       color=self.colors[i], linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUROC')
        ax.set_title('Aggregation Method Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_training_stability(self, ax):
        """Plot training stability metrics."""
        # Plot learning rate and accuracy
        train_acc = self.metrics_history.get('train/accuracy', [])
        val_acc = self.metrics_history.get('val/accuracy', [])
        
        if train_acc:
            epochs = self.epoch_history[:len(train_acc)]
            ax.plot(epochs, train_acc, label='Train Accuracy', 
                   color=self.colors[0], linewidth=2)
        
        if val_acc:
            epochs = self.epoch_history[:len(val_acc)]
            ax.plot(epochs, val_acc, label='Val Accuracy', 
                   color=self.colors[1], linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Stability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _calculate_simple_auroc(self, probs, targets):
        """Simple AUROC calculation."""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(targets, probs)
        except ImportError:
            # Fallback: simple approximation
            return 0.5  # Placeholder
        except:
            return np.nan
    
    def save_metrics_history(self):
        """Save metrics history to JSON for later analysis."""
        history_path = self.plots_dir / "metrics_history.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, values in self.metrics_history.items():
            serializable_history[key] = [float(v) if not np.isnan(v) else None for v in values]
        
        with open(history_path, 'w') as f:
            json.dump({
                'epochs': self.epoch_history,
                'metrics': serializable_history,
                'subject_results_count': len(self.subject_results_history)
            }, f, indent=2)
    
    def generate_final_summary(self):
        """Generate final training summary plots."""
        if not self.enable_live:
            return
            
        try:
            # Create a comprehensive final summary
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss summary
            self._plot_loss_curves(axes[0, 0])
            
            # AUROC summary
            self._plot_auroc_comparison(axes[0, 1])
            
            # Method ranking summary
            self._plot_method_ranking(axes[1, 0])
            
            # Best epoch analysis
            ax = axes[1, 1]
            auroc_values = self.metrics_history.get('val/auroc_subject', [])
            if auroc_values:
                best_epoch = np.argmax(auroc_values)
                best_value = auroc_values[best_epoch]
                
                ax.bar(['Best AUROC', 'Final AUROC'], 
                      [best_value, auroc_values[-1] if auroc_values else 0],
                      color=[self.colors[0], self.colors[1]])
                ax.set_ylabel('AUROC')
                ax.set_title(f'Performance Summary\nBest: {best_value:.3f} at epoch {best_epoch}')
                ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.suptitle('Training Summary', fontsize=16, fontweight='bold', y=1.02)
            
            summary_path = self.plots_dir / "training_summary.png"
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Training summary saved to {summary_path}")
            
        except Exception as e:
            print(f"Error generating training summary: {e}")
            plt.close('all')


class LiveTrainingMonitor:
    """
    Lightweight monitor that can be used as a Lightning callback
    to integrate with the training process.
    """
    
    def __init__(self, visualizer: TrainingVisualizer):
        self.visualizer = visualizer
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of validation epoch."""
        # Extract metrics from trainer
        metrics = trainer.logged_metrics
        epoch = trainer.current_epoch
        
        # Convert tensor metrics to float
        float_metrics = {}
        for key, value in metrics.items():
            try:
                if hasattr(value, 'item'):
                    float_metrics[key] = value.item()
                else:
                    float_metrics[key] = float(value)
            except:
                continue
        
        # Get subject results if available
        subject_results = None
        if hasattr(pl_module, '_last_subject_results'):
            subject_results = pl_module._last_subject_results
        
        # Update visualizer
        self.visualizer.update_metrics(epoch, float_metrics, subject_results)
    
    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        self.visualizer.save_metrics_history()
        self.visualizer.generate_final_summary()
