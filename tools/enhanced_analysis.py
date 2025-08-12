"""
Enhanced analysis tool for comparing aggregation methods and generating comprehensive reports.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from collections import defaultdict

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedTrainingAnalyzer:
    """
    Comprehensive analyzer for enhanced training results with multiple aggregation methods.
    """
    
    def __init__(self, base_results_dir: str):
        """
        Initialize analyzer.
        
        Args:
            base_results_dir: Base directory containing fold results
        """
        self.base_dir = Path(base_results_dir)
        self.fold_results = {}
        self.aggregation_methods = []
        self.method_performance = defaultdict(list)
        
    def load_all_fold_results(self, num_folds: int = 3):
        """Load results from all folds."""
        print(f"Loading results from {num_folds} folds...")
        
        for fold_idx in range(num_folds):
            fold_dir = self.base_dir / f"fold{fold_idx}"
            self.load_fold_results(fold_idx, fold_dir)
        
        # Extract common aggregation methods
        if self.fold_results:
            first_fold = list(self.fold_results.values())[0]
            if 'enhanced_results' in first_fold:
                sample_result = first_fold['enhanced_results'][0]
                if 'method_aurocs' in sample_result:
                    self.aggregation_methods = list(sample_result['method_aurocs'].keys())
        
        print(f"Found aggregation methods: {self.aggregation_methods}")
        
    def load_fold_results(self, fold_idx: int, fold_dir: Path):
        """Load results from a single fold."""
        if not fold_dir.exists():
            print(f"Warning: Fold directory {fold_dir} does not exist")
            return
        
        # Look for task directory
        task_dirs = list(fold_dir.glob("Task*"))
        if not task_dirs:
            print(f"Warning: No task directory found in {fold_dir}")
            return
        
        task_dir = task_dirs[0]
        version_dirs = list(task_dir.glob("unet_xl/version_*"))
        if not version_dirs:
            print(f"Warning: No version directory found in {task_dir}")
            return
        
        version_dir = version_dirs[0]
        
        fold_data = {
            'fold_idx': fold_idx,
            'version_dir': version_dir,
            'training_logs': [],
            'subject_probs': [],
            'enhanced_results': [],
            'aggregation_results': []
        }
        
        # Load training logs
        training_log_path = version_dir / "training_log.txt"
        if training_log_path.exists():
            fold_data['training_logs'] = self.parse_training_log(training_log_path)
        
        # Load subject probabilities
        subject_probs_dir = version_dir / "subject_probs"
        if subject_probs_dir.exists():
            fold_data['subject_probs'] = self.load_subject_probs(subject_probs_dir)
        
        # Load enhanced results
        enhanced_dir = subject_probs_dir
        if enhanced_dir and enhanced_dir.exists():
            fold_data['enhanced_results'] = self.load_enhanced_results(enhanced_dir)
        
        # Load aggregation results
        aggregation_dir = version_dir / "aggregation_results"
        if aggregation_dir.exists():
            fold_data['aggregation_results'] = self.load_aggregation_results(aggregation_dir)
        
        self.fold_results[fold_idx] = fold_data
        print(f"Loaded fold {fold_idx} from {version_dir}")
    
    def parse_training_log(self, log_path: Path) -> List[Dict]:
        """Parse training log file."""
        logs = []
        current_epoch = None
        current_entry = {}
        
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if 'Current Epoch:' in line:
                    if current_entry:
                        logs.append(current_entry)
                    current_epoch = int(line.split()[-1])
                    current_entry = {'epoch': current_epoch}
                elif ':' in line and current_epoch is not None:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        try:
                            value = float(parts[1].strip())
                            current_entry[key] = value
                        except ValueError:
                            current_entry[key] = parts[1].strip()
        
        if current_entry:
            logs.append(current_entry)
        
        return logs
    
    def load_subject_probs(self, subject_probs_dir: Path) -> List[Dict]:
        """Load subject probability files."""
        results = []
        json_files = list(subject_probs_dir.glob("val_subject_probs_epoch_*.json"))
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return results
    
    def load_enhanced_results(self, enhanced_dir: Path) -> List[Dict]:
        """Load enhanced aggregation results."""
        results = []
        enhanced_files = list(enhanced_dir.glob("val_subject_probs_enhanced_epoch_*.json"))
        
        for enhanced_file in sorted(enhanced_files):
            try:
                with open(enhanced_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {enhanced_file}: {e}")
        
        return results
    
    def load_aggregation_results(self, aggregation_dir: Path) -> List[Dict]:
        """Load aggregation analysis results."""
        results = []
        agg_files = list(aggregation_dir.glob("aggregation_epoch_*.json"))
        
        for agg_file in sorted(agg_files):
            try:
                with open(agg_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {agg_file}: {e}")
        
        return results
    
    def compute_method_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Compute performance summary for all aggregation methods."""
        method_summary = {}
        
        # Collect all AUROC values per method across folds
        for fold_idx, fold_data in self.fold_results.items():
            if 'enhanced_results' in fold_data:
                for result in fold_data['enhanced_results']:
                    if 'method_aurocs' in result:
                        for method, auroc in result['method_aurocs'].items():
                            if not np.isnan(auroc):
                                self.method_performance[method].append(auroc)
        
        # Compute summary statistics
        for method, aurocs in self.method_performance.items():
            if aurocs:
                method_summary[method] = {
                    'mean_auroc': np.mean(aurocs),
                    'std_auroc': np.std(aurocs),
                    'min_auroc': np.min(aurocs),
                    'max_auroc': np.max(aurocs),
                    'median_auroc': np.median(aurocs),
                    'n_evaluations': len(aurocs)
                }
        
        return method_summary
    
    def generate_comprehensive_report(self, output_dir: str):
        """Generate comprehensive analysis report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating comprehensive report in {output_path}")
        
        # 1. Method performance comparison
        self.plot_method_performance_comparison(output_path / "method_performance_comparison.png")
        
        # 2. Training dynamics per fold
        self.plot_training_dynamics_per_fold(output_path / "training_dynamics_per_fold.png")
        
        # 3. Method ranking over time
        self.plot_method_ranking_over_time(output_path / "method_ranking_over_time.png")
        
        # 4. Cross-fold stability analysis
        self.plot_cross_fold_stability(output_path / "cross_fold_stability.png")
        
        # 5. Generate text summary
        self.generate_text_summary(output_path / "analysis_summary.txt")
        
        # 6. Export detailed results
        self.export_detailed_results(output_path / "detailed_results.json")
        
        print("Comprehensive report generated successfully!")
    
    def plot_method_performance_comparison(self, output_path: Path):
        """Plot comparison of aggregation method performance."""
        method_summary = self.compute_method_performance_summary()
        
        if not method_summary:
            print("No method performance data available")
            return
        
        methods = list(method_summary.keys())
        means = [method_summary[m]['mean_auroc'] for m in methods]
        stds = [method_summary[m]['std_auroc'] for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot with error bars
        colors = sns.color_palette("husl", len(methods))
        bars = ax1.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax1.set_title('Aggregation Method Performance Comparison')
        ax1.set_ylabel('AUROC')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
        
        # Box plot showing distribution
        method_data = []
        method_labels = []
        for method in methods:
            if method in self.method_performance:
                method_data.append(self.method_performance[method])
                method_labels.append(method)
        
        if method_data:
            ax2.boxplot(method_data, labels=method_labels)
            ax2.set_title('AUROC Distribution per Method')
            ax2.set_ylabel('AUROC')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_dynamics_per_fold(self, output_path: Path):
        """Plot training dynamics for each fold."""
        n_folds = len(self.fold_results)
        if n_folds == 0:
            return
        
        fig, axes = plt.subplots(n_folds, 2, figsize=(15, 4 * n_folds))
        if n_folds == 1:
            axes = axes.reshape(1, -1)
        
        for fold_idx, fold_data in self.fold_results.items():
            row = fold_idx
            
            # Plot training metrics
            if 'training_logs' in fold_data and fold_data['training_logs']:
                logs = fold_data['training_logs']
                epochs = [log.get('epoch', 0) for log in logs]
                
                # Loss curves
                train_losses = [log.get('train/loss', np.nan) for log in logs]
                val_losses = [log.get('val/loss', np.nan) for log in logs]
                
                axes[row, 0].plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
                axes[row, 0].plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
                axes[row, 0].set_title(f'Fold {fold_idx}: Loss Curves')
                axes[row, 0].set_xlabel('Epoch')
                axes[row, 0].set_ylabel('Loss')
                axes[row, 0].legend()
                axes[row, 0].grid(True, alpha=0.3)
                
                # AUROC curves
                val_aurocs = [log.get('val/auroc_subject', np.nan) for log in logs]
                axes[row, 1].plot(epochs, val_aurocs, label='Val AUROC', marker='o', markersize=3)
                axes[row, 1].set_title(f'Fold {fold_idx}: Validation AUROC')
                axes[row, 1].set_xlabel('Epoch')
                axes[row, 1].set_ylabel('AUROC')
                axes[row, 1].legend()
                axes[row, 1].grid(True, alpha=0.3)
                axes[row, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_method_ranking_over_time(self, output_path: Path):
        """Plot how method rankings change over epochs."""
        if not self.aggregation_methods:
            print("No aggregation methods found")
            return
        
        # Collect method performance over epochs across all folds
        epoch_method_performance = defaultdict(lambda: defaultdict(list))
        
        for fold_idx, fold_data in self.fold_results.items():
            if 'enhanced_results' in fold_data:
                for result in fold_data['enhanced_results']:
                    epoch = result.get('epoch', 0)
                    if 'method_aurocs' in result:
                        for method, auroc in result['method_aurocs'].items():
                            if not np.isnan(auroc):
                                epoch_method_performance[epoch][method].append(auroc)
        
        # Average across folds for each epoch
        epochs = sorted(epoch_method_performance.keys())
        method_curves = {method: [] for method in self.aggregation_methods}
        
        for epoch in epochs:
            for method in self.aggregation_methods:
                if method in epoch_method_performance[epoch]:
                    avg_auroc = np.mean(epoch_method_performance[epoch][method])
                    method_curves[method].append(avg_auroc)
                else:
                    method_curves[method].append(np.nan)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = sns.color_palette("husl", len(self.aggregation_methods))
        
        for i, method in enumerate(self.aggregation_methods):
            values = method_curves[method]
            valid_epochs = []
            valid_values = []
            
            for epoch, value in zip(epochs, values):
                if not np.isnan(value):
                    valid_epochs.append(epoch)
                    valid_values.append(value)
            
            if valid_values:
                ax.plot(valid_epochs, valid_values, label=method, 
                       color=colors[i], marker='o', markersize=4, linewidth=2)
        
        ax.set_title('Aggregation Method Performance Over Training')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average AUROC Across Folds')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cross_fold_stability(self, output_path: Path):
        """Plot cross-fold stability analysis."""
        method_summary = self.compute_method_performance_summary()
        
        if not method_summary:
            return
        
        methods = list(method_summary.keys())
        means = [method_summary[m]['mean_auroc'] for m in methods]
        stds = [method_summary[m]['std_auroc'] for m in methods]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot: mean vs std (stability)
        colors = sns.color_palette("husl", len(methods))
        scatter = ax.scatter(stds, means, c=colors, s=100, alpha=0.7)
        
        # Add method labels
        for i, method in enumerate(methods):
            ax.annotate(method, (stds[i], means[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Standard Deviation (Instability)')
        ax.set_ylabel('Mean AUROC (Performance)')
        ax.set_title('Method Performance vs Stability\n(Top-right quadrant = high performance + stable)')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant lines
        mean_std = np.mean(stds)
        mean_auroc = np.mean(means)
        ax.axvline(mean_std, color='red', linestyle='--', alpha=0.5)
        ax.axhline(mean_auroc, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_text_summary(self, output_path: Path):
        """Generate text summary of analysis."""
        method_summary = self.compute_method_performance_summary()
        
        with open(output_path, 'w') as f:
            f.write("ENHANCED TRAINING ANALYSIS SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Number of folds analyzed: {len(self.fold_results)}\n")
            f.write(f"Aggregation methods found: {len(self.aggregation_methods)}\n")
            f.write(f"Methods: {', '.join(self.aggregation_methods)}\n\n")
            
            if method_summary:
                f.write("METHOD PERFORMANCE SUMMARY\n")
                f.write("-" * 30 + "\n")
                
                # Sort methods by mean performance
                sorted_methods = sorted(method_summary.items(), 
                                      key=lambda x: x[1]['mean_auroc'], reverse=True)
                
                for rank, (method, stats) in enumerate(sorted_methods, 1):
                    f.write(f"{rank}. {method}\n")
                    f.write(f"   Mean AUROC: {stats['mean_auroc']:.4f} ± {stats['std_auroc']:.4f}\n")
                    f.write(f"   Range: [{stats['min_auroc']:.4f}, {stats['max_auroc']:.4f}]\n")
                    f.write(f"   Median: {stats['median_auroc']:.4f}\n")
                    f.write(f"   Evaluations: {stats['n_evaluations']}\n\n")
                
                # Best method recommendation
                best_method = sorted_methods[0][0]
                best_stats = sorted_methods[0][1]
                
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 15 + "\n")
                f.write(f"Best performing method: {best_method}\n")
                f.write(f"Performance: {best_stats['mean_auroc']:.4f} ± {best_stats['std_auroc']:.4f}\n\n")
                
                # Stability analysis
                most_stable = min(sorted_methods, key=lambda x: x[1]['std_auroc'])
                f.write(f"Most stable method: {most_stable[0]}\n")
                f.write(f"Stability (std): {most_stable[1]['std_auroc']:.4f}\n\n")
            
            # Training insights
            f.write("TRAINING INSIGHTS\n")
            f.write("-" * 17 + "\n")
            
            total_epochs = 0
            for fold_data in self.fold_results.values():
                if 'training_logs' in fold_data:
                    total_epochs += len(fold_data['training_logs'])
            
            avg_epochs_per_fold = total_epochs / len(self.fold_results) if self.fold_results else 0
            f.write(f"Average epochs per fold: {avg_epochs_per_fold:.1f}\n")
            
            f.write("\nThis analysis was generated by EnhancedTrainingAnalyzer\n")
    
    def export_detailed_results(self, output_path: Path):
        """Export detailed results to JSON."""
        detailed_results = {
            'fold_results': {},
            'method_performance_summary': self.compute_method_performance_summary(),
            'aggregation_methods': self.aggregation_methods,
            'analysis_metadata': {
                'num_folds': len(self.fold_results),
                'total_methods': len(self.aggregation_methods)
            }
        }
        
        # Convert fold results to serializable format
        for fold_idx, fold_data in self.fold_results.items():
            detailed_results['fold_results'][str(fold_idx)] = {
                'fold_idx': fold_data['fold_idx'],
                'num_training_logs': len(fold_data.get('training_logs', [])),
                'num_subject_probs': len(fold_data.get('subject_probs', [])),
                'num_enhanced_results': len(fold_data.get('enhanced_results', [])),
                'num_aggregation_results': len(fold_data.get('aggregation_results', []))
            }
        
        with open(output_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Enhanced training analysis tool")
    parser.add_argument("results_dir", help="Base directory containing fold results")
    parser.add_argument("--output_dir", default="./analysis_output", 
                       help="Output directory for analysis results")
    parser.add_argument("--num_folds", type=int, default=3, 
                       help="Number of folds to analyze")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = EnhancedTrainingAnalyzer(args.results_dir)
    
    # Load results
    analyzer.load_all_fold_results(args.num_folds)
    
    # Generate report
    analyzer.generate_comprehensive_report(args.output_dir)
    
    print(f"Analysis complete! Check {args.output_dir} for results.")


if __name__ == "__main__":
    main()
