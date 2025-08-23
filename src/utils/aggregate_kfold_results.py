#!/usr/bin/env python3
"""
K-Fold Cross-Validation Results Aggregation Utility

Aggregates metrics across K folds and computes statistics for FOMO Task3.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def extract_metrics_from_tensorboard_logs(experiment_dir: str, target_metrics: List[str]) -> Dict[str, float]:
    """
    Extract best metrics from TensorBoard logs using wandb or PyTorch Lightning logs.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., version_0)
        target_metrics: List of metric names to extract (e.g., ['val/corr', 'val/mae'])
        
    Returns:
        Dictionary of metric_name -> best_value
    """
    metrics_dict = {}
    
    # Look for metrics.csv (PyTorch Lightning default)
    metrics_csv = os.path.join(experiment_dir, "metrics.csv")
    
    if os.path.exists(metrics_csv):
        logger.debug(f"Loading metrics from: {metrics_csv}")
        try:
            df = pd.read_csv(metrics_csv)
            
            for metric in target_metrics:
                if metric in df.columns:
                    # Get best value (max for correlation, min for MAE/loss)
                    valid_values = df[metric].dropna()
                    if len(valid_values) > 0:
                        if 'corr' in metric.lower() or 'r2' in metric.lower():
                            # Higher is better for correlation/R²
                            metrics_dict[metric] = valid_values.max()
                        else:
                            # Lower is better for MAE/MSE/loss
                            metrics_dict[metric] = valid_values.min()
                        
                        logger.debug(f"  {metric}: {metrics_dict[metric]:.4f}")
        
        except Exception as e:
            logger.warning(f"Failed to read metrics.csv: {e}")
    
    # Also try to extract from hparams.yaml if available
    hparams_file = os.path.join(experiment_dir, "hparams.yaml")
    if os.path.exists(hparams_file) and not metrics_dict:
        logger.debug(f"Trying hparams.yaml: {hparams_file}")
        # This is a fallback - usually metrics aren't in hparams
        
    return metrics_dict


def find_experiment_directories(results_dir: str, experiment_pattern: str) -> List[str]:
    """
    Find all experiment directories matching the pattern.
    
    Args:
        results_dir: Base results directory (e.g., ./runs/Task003_FOMO3/unet_xl)
        experiment_pattern: Pattern to match experiment names (e.g., "fomo3_brain_age_256_kfold_fold*")
        
    Returns:
        List of paths to experiment directories
    """
    # Look for directories matching the pattern
    pattern_path = os.path.join(results_dir, experiment_pattern, "version_*")
    matched_dirs = glob.glob(pattern_path)
    
    # Filter to only include directories with version_X structure
    experiment_dirs = []
    for dir_path in matched_dirs:
        if os.path.isdir(dir_path) and "version_" in os.path.basename(dir_path):
            experiment_dirs.append(dir_path)
    
    # Sort by fold number for consistent ordering
    experiment_dirs.sort(key=lambda x: extract_fold_number(x))
    
    logger.info(f"Found {len(experiment_dirs)} experiment directories:")
    for exp_dir in experiment_dirs:
        logger.info(f"  {exp_dir}")
    
    return experiment_dirs


def extract_fold_number(experiment_dir: str) -> int:
    """Extract fold number from experiment directory path."""
    # Look for fold number in path: ...fomo3_brain_age_256_kfold_fold3/version_0
    parts = experiment_dir.split('/')
    for part in parts:
        if 'fold' in part:
            # Extract number after 'fold'
            try:
                fold_num = ''.join(filter(str.isdigit, part.split('fold')[-1]))
                if fold_num:
                    return int(fold_num)
            except (ValueError, IndexError):
                continue
    return 0  # Default fallback


def aggregate_kfold_metrics(experiment_dirs: List[str], target_metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across all K-fold experiment directories.
    
    Args:
        experiment_dirs: List of experiment directory paths
        target_metrics: List of metric names to aggregate
        
    Returns:
        Dictionary with aggregated statistics per metric
    """
    # Collect metrics from all folds
    fold_metrics = {}  # fold_idx -> {metric: value}
    
    for exp_dir in experiment_dirs:
        fold_idx = extract_fold_number(exp_dir)
        metrics = extract_metrics_from_tensorboard_logs(exp_dir, target_metrics)
        
        if metrics:
            fold_metrics[fold_idx] = metrics
            logger.info(f"Fold {fold_idx}: {metrics}")
        else:
            logger.warning(f"No metrics found for fold {fold_idx} in {exp_dir}")
    
    # Compute aggregated statistics
    aggregated = {}
    
    for metric in target_metrics:
        values = []
        for fold_idx, fold_data in fold_metrics.items():
            if metric in fold_data:
                values.append(fold_data[metric])
        
        if values:
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'values': values,
                'n_folds': len(values)
            }
            
            logger.info(f"{metric}: {aggregated[metric]['mean']:.4f} ± {aggregated[metric]['std']:.4f} "
                       f"(range: {aggregated[metric]['min']:.4f}-{aggregated[metric]['max']:.4f})")
        else:
            logger.warning(f"No values found for metric: {metric}")
    
    return aggregated


def save_aggregated_results(aggregated_metrics: Dict[str, Dict[str, float]], 
                           output_file: str, 
                           fold_metrics: Optional[Dict[int, Dict[str, float]]] = None) -> None:
    """
    Save aggregated K-fold results to a summary file.
    
    Args:
        aggregated_metrics: Aggregated statistics per metric
        output_file: Path to output summary file
        fold_metrics: Optional individual fold metrics for detailed reporting
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("FOMO Task 3 - 5-Fold Cross-Validation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        # Summary statistics
        f.write("AGGREGATED METRICS SUMMARY\n")
        f.write("-" * 30 + "\n")
        
        for metric, stats in aggregated_metrics.items():
            metric_display = metric.replace('val/', '').upper()
            f.write(f"{metric_display}:\n")
            f.write(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n")
            f.write(f"  Median: {stats['median']:.4f}\n")
            f.write(f"  Folds: {stats['n_folds']}/5\n")
            f.write("\n")
        
        # Individual fold details
        if fold_metrics:
            f.write("INDIVIDUAL FOLD RESULTS\n")
            f.write("-" * 25 + "\n")
            
            for fold_idx in sorted(fold_metrics.keys()):
                fold_data = fold_metrics[fold_idx]
                f.write(f"Fold {fold_idx}:\n")
                for metric, value in fold_data.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        
        # Performance interpretation
        f.write("PERFORMANCE INTERPRETATION\n")
        f.write("-" * 28 + "\n")
        
        if 'val/corr' in aggregated_metrics:
            corr_mean = aggregated_metrics['val/corr']['mean']
            if corr_mean >= 0.8:
                perf_level = "Excellent"
            elif corr_mean >= 0.7:
                perf_level = "Good"
            elif corr_mean >= 0.6:
                perf_level = "Moderate"
            else:
                perf_level = "Poor"
            
            f.write(f"Brain Age Prediction Performance: {perf_level}\n")
            f.write(f"  Correlation: {corr_mean:.3f} (>0.8=excellent, >0.7=good, >0.6=moderate)\n")
        
        if 'val/mae' in aggregated_metrics:
            mae_mean = aggregated_metrics['val/mae']['mean']
            f.write(f"  Mean Absolute Error: {mae_mean:.2f} years\n")
            
            if mae_mean <= 3.0:
                acc_level = "High"
            elif mae_mean <= 5.0:
                acc_level = "Moderate"  
            else:
                acc_level = "Low"
            f.write(f"  Accuracy Level: {acc_level} (<3y=high, <5y=moderate)\n")
    
    logger.info(f"Results summary saved to: {output_file}")


def main():
    """Command-line interface for K-fold results aggregation."""
    
    parser = argparse.ArgumentParser(
        description="Aggregate K-fold cross-validation results for FOMO Task3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python aggregate_kfold_results.py \\
    --results_dir ./runs/Task003_FOMO3/unet_xl \\
    --experiment_pattern "fomo3_brain_age_256_kfold_fold*" \\
    --metrics val/corr,val/mae

  # With custom output
  python aggregate_kfold_results.py \\
    --results_dir ./runs/Task003_FOMO3/unet_xl \\
    --experiment_pattern "fomo3_*_kfold_fold*" \\
    --output_file ./custom_kfold_summary.txt \\
    --metrics val/corr,val/mae,val/mse
        """
    )
    
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Base results directory (e.g., ./runs/Task003_FOMO3/unet_xl)")
    parser.add_argument("--experiment_pattern", type=str, required=True,
                       help="Pattern to match experiment names (e.g., fomo3_brain_age_256_kfold_fold*)")
    parser.add_argument("--metrics", type=str, default="val/corr,val/mae",
                       help="Comma-separated list of metrics to aggregate")
    parser.add_argument("--output_file", type=str, 
                       default="./runs/Task003_FOMO3/kfold_results_summary.txt",
                       help="Output file for aggregated results summary")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(levelname)s:%(name)s:%(message)s')
    
    try:
        # Parse metrics list
        target_metrics = [m.strip() for m in args.metrics.split(',')]
        logger.info(f"Target metrics: {target_metrics}")
        
        # Find experiment directories
        print("🔍 Finding experiment directories...")
        experiment_dirs = find_experiment_directories(args.results_dir, args.experiment_pattern)
        
        if not experiment_dirs:
            print(f"❌ No experiment directories found matching pattern: {args.experiment_pattern}")
            print(f"   in directory: {args.results_dir}")
            return 1
        
        print(f"✅ Found {len(experiment_dirs)} experiment directories")
        
        # Aggregate metrics
        print("📊 Aggregating metrics across folds...")
        aggregated_metrics = aggregate_kfold_metrics(experiment_dirs, target_metrics)
        
        if not aggregated_metrics:
            print("❌ No metrics could be aggregated")
            return 1
        
        # Extract fold metrics for detailed reporting
        fold_metrics = {}
        for exp_dir in experiment_dirs:
            fold_idx = extract_fold_number(exp_dir)
            metrics = extract_metrics_from_tensorboard_logs(exp_dir, target_metrics)
            if metrics:
                fold_metrics[fold_idx] = metrics
        
        # Save results
        print("💾 Saving aggregated results...")
        save_aggregated_results(aggregated_metrics, args.output_file, fold_metrics)
        
        # Print summary
        print("\n🎉 K-Fold aggregation completed!")
        print("=" * 40)
        for metric, stats in aggregated_metrics.items():
            print(f"{metric.replace('val/', '').upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"📄 Full results: {args.output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
