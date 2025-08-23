#!/usr/bin/env python3
"""
Utility to extract and save metrics information from existing checkpoints.
This script creates a metrics.txt file for existing checkpoints that may be missing this information.

Usage:
    python src/utils/extract_checkpoint_metrics.py --checkpoint_path /path/to/best.ckpt
    
The script will:
1. Load the checkpoint and extract available metrics
2. Create a comprehensive metrics.txt file with:
   - Best validation metrics at the checkpoint epoch
   - Full epoch information including training and validation statistics
   - Timestamp and configuration information
"""

import argparse
import os
import sys
import torch
import yaml
from datetime import datetime
from pathlib import Path

def extract_metrics_from_checkpoint(checkpoint_path: str, output_dir: str = None) -> str:
    """
    Extract metrics from a PyTorch Lightning checkpoint and save to metrics.txt
    
    Args:
        checkpoint_path: Path to the .ckpt file
        output_dir: Directory to save metrics.txt (defaults to same directory as checkpoint)
        
    Returns:
        Path to the created metrics.txt file
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
    
    # Extract basic information
    epoch = checkpoint.get('epoch', 'Unknown')
    global_step = checkpoint.get('global_step', 'Unknown')
    
    # Look for hparams.yaml in the same directory structure
    checkpoint_dir = os.path.dirname(checkpoint_path)
    version_dir = os.path.dirname(checkpoint_dir)
    hparams_path = os.path.join(version_dir, 'hparams.yaml')
    
    hparams = {}
    if os.path.exists(hparams_path):
        try:
            with open(hparams_path, 'r') as f:
                hparams = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load hparams.yaml: {e}")
    
    # Extract task information
    task_info = {}
    if 'config' in hparams:
        config = hparams['config']
        task_info = {
            'task_name': config.get('task', 'Unknown'),
            'task_type': config.get('task_type', 'Unknown'),
            'experiment': config.get('experiment', 'Unknown'),
            'model_name': config.get('model_name', 'Unknown'),
        }
    
    # Try to extract metrics from various checkpoint fields
    metrics = {}
    
    # Look for logged metrics in different checkpoint fields
    possible_metric_fields = [
        'lr_schedulers', 
        'epoch', 
        'global_step', 
        'pytorch-lightning_version',
        'state_dict',
        'optimizer_states',
        'callbacks'
    ]
    
    # Check for callback state that might contain metrics
    if 'callbacks' in checkpoint:
        callbacks = checkpoint['callbacks']
        for callback_name, callback_state in callbacks.items():
            if 'best_model_score' in callback_state:
                metrics['best_score'] = float(callback_state['best_model_score'])
            if 'best_model_path' in callback_state:
                metrics['best_model_path'] = callback_state['best_model_path']
    
    # Create metrics.txt content
    metrics_content = []
    metrics_content.append("# FOMO Checkpoint Metrics Information")
    metrics_content.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    metrics_content.append(f"# Checkpoint: {os.path.basename(checkpoint_path)}")
    metrics_content.append("")
    
    # Basic checkpoint information
    metrics_content.append("## Checkpoint Information")
    metrics_content.append(f"Epoch: {epoch}")
    metrics_content.append(f"Global Step: {global_step}")
    metrics_content.append("")
    
    # Task information
    if task_info:
        metrics_content.append("## Task Configuration")
        for key, value in task_info.items():
            metrics_content.append(f"{key.replace('_', ' ').title()}: {value}")
        metrics_content.append("")
    
    # Model metrics (if available)
    if metrics:
        metrics_content.append("## Best Metrics")
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_content.append(f"{key}: {value:.6f}")
            else:
                metrics_content.append(f"{key}: {value}")
        metrics_content.append("")
    
    # Training configuration (from hparams)
    if 'config' in hparams:
        config = hparams['config']
        metrics_content.append("## Training Configuration")
        
        # Key training parameters
        key_params = [
            'learning_rate', 'batch_size', 'epochs', 'loss_type', 
            'age_normalization', 'age_mean', 'age_std', 'patch_size',
            'precision', 'augmentation_preset'
        ]
        
        for param in key_params:
            if param in config:
                value = config[param]
                metrics_content.append(f"{param}: {value}")
        
        metrics_content.append("")
    
    # Expected metrics based on task type
    task_type = task_info.get('task_type', 'unknown').lower()
    metrics_content.append("## Expected Metrics (Task-Specific)")
    
    if task_type == 'regression':
        metrics_content.append("Primary Monitor Metric: val/corr (Pearson Correlation)")
        metrics_content.append("Secondary Metrics:")
        metrics_content.append("  - val/mae: Mean Absolute Error (years)")
        metrics_content.append("  - val/loss: Training loss (normalized)")
        metrics_content.append("  - train/corr: Training correlation")
        metrics_content.append("  - train/mae: Training MAE")
    elif task_type == 'classification':
        metrics_content.append("Primary Monitor Metric: val/auroc_subject")
        metrics_content.append("Secondary Metrics:")
        metrics_content.append("  - val/accuracy: Classification accuracy")
        metrics_content.append("  - val/f1: F1 score")
        metrics_content.append("  - val/precision: Precision")
        metrics_content.append("  - val/recall: Recall")
    elif task_type == 'segmentation':
        metrics_content.append("Primary Monitor Metric: val/loss")
        metrics_content.append("Secondary Metrics:")
        metrics_content.append("  - val/dice: Dice coefficient")
        metrics_content.append("  - val/nsd: Normal Surface Distance")
    
    metrics_content.append("")
    
    # Note about metrics extraction
    metrics_content.append("## Notes")
    metrics_content.append("- This file was generated from checkpoint metadata")
    metrics_content.append("- For complete metrics history, refer to training logs")
    metrics_content.append("- Best metrics are recorded when model performance improves")
    metrics_content.append(f"- Checkpoint saved at epoch {epoch}, step {global_step}")
    
    # Write metrics.txt file
    metrics_file_path = os.path.join(output_dir, 'metrics.txt')
    try:
        with open(metrics_file_path, 'w') as f:
            f.write('\n'.join(metrics_content))
        
        print(f"✅ Created metrics file: {metrics_file_path}")
        return metrics_file_path
        
    except Exception as e:
        print(f"❌ Error writing metrics file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Extract metrics information from FOMO checkpoint files"
    )
    parser.add_argument(
        '--checkpoint_path', 
        required=True,
        help='Path to the checkpoint file (best.ckpt)'
    )
    parser.add_argument(
        '--output_dir',
        help='Directory to save metrics.txt (default: same as checkpoint)'
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    if not args.checkpoint_path.endswith('.ckpt'):
        print(f"⚠️  Warning: File doesn't have .ckpt extension: {args.checkpoint_path}")
    
    # Extract metrics
    result = extract_metrics_from_checkpoint(args.checkpoint_path, args.output_dir)
    
    if result:
        print(f"✅ Successfully created metrics file")
        print(f"📁 Location: {result}")
    else:
        print(f"❌ Failed to create metrics file")
        sys.exit(1)

if __name__ == "__main__":
    main()
