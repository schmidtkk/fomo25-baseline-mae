#!/usr/bin/env python
"""
Task3 Brain Age Regression Inference Script

Predicts brain age from T1 and T2 MRI modalities using trained Task3 models.
Supports automatic checkpoint detection, age denormalization, and test-time augmentation.
"""

import argparse
import os
import sys
import torch
import yaml
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference.predict import predict_from_config, ModelLoader, save_output_txt
from data.task_configs import task3_config


def load_task3_config(config_path: str) -> Dict[str, Any]:
    """
    Load and process Task3 hyperparameters from hparams.yaml.
    
    Args:
        config_path: Path to hparams.yaml file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            hparams = yaml.safe_load(f)
        
        if 'config' not in hparams:
            raise KeyError("'config' key not found in hparams.yaml")
            
        return hparams['config']
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML format in {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading config from {config_path}: {e}")


def denormalize_age(normalized_age: float, age_mean: float, age_std: float) -> float:
    """
    Convert normalized age prediction back to years.
    
    Args:
        normalized_age: Age prediction on normalized scale
        age_mean: Mean age used for normalization during training
        age_std: Standard deviation used for normalization during training
        
    Returns:
        Age in years
    """
    return float(normalized_age * age_std + age_mean)


def validate_modality_files(modalities: list) -> None:
    """
    Validate that modality files exist and are readable.
    
    Args:
        modalities: List of modality file paths
        
    Raises:
        FileNotFoundError: If any modality file doesn't exist
        ValueError: If wrong number of modalities provided
    """
    if len(modalities) != 2:
        raise ValueError(f"Task3 requires exactly 2 modalities (T1 and T2), got {len(modalities)}")
    
    for i, modality_path in enumerate(modalities):
        if not os.path.exists(modality_path):
            raise FileNotFoundError(f"Modality {i+1} file not found: {modality_path}")
        
        if not modality_path.lower().endswith(('.nii', '.nii.gz')):
            logging.warning(f"Modality {i+1} doesn't have .nii/.nii.gz extension: {modality_path}")


def select_device(device_arg: str) -> torch.device:
    """
    Select appropriate device for inference.
    
    Args:
        device_arg: Device argument ('auto', 'cpu', 'cuda', 'cuda:X')
        
    Returns:
        PyTorch device
    """
    if device_arg.lower() == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info(f"Auto-selected device: {device} (GPU available)")
        else:
            device = torch.device('cpu')
            logging.info(f"Auto-selected device: {device} (no GPU available)")
    else:
        device = torch.device(device_arg)
        logging.info(f"Using specified device: {device}")
    
    return device


def find_default_checkpoint() -> str:
    """
    Find the default Task3 checkpoint if none specified.
    
    Returns:
        Path to best checkpoint
        
    Raises:
        FileNotFoundError: If no checkpoint found
    """
    default_path = "runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/checkpoints/best.ckpt"
    
    if os.path.exists(default_path):
        return default_path
    
    # Search for alternative checkpoints
    runs_dir = "runs/Task003_FOMO3/unet_xl"
    if os.path.exists(runs_dir):
        for experiment_dir in os.listdir(runs_dir):
            checkpoint_path = os.path.join(runs_dir, experiment_dir, "version_0", "checkpoints", "best.ckpt")
            if os.path.exists(checkpoint_path):
                logging.warning(f"Using alternative checkpoint: {checkpoint_path}")
                return checkpoint_path
    
    raise FileNotFoundError(
        f"No Task3 checkpoint found. Please train a model first or specify --checkpoint_path"
    )


def find_default_config() -> str:
    """
    Find the default Task3 config if none specified.
    
    Returns:
        Path to hparams.yaml
        
    Raises:
        FileNotFoundError: If no config found
    """
    default_path = "runs/Task003_FOMO3/unet_xl/fomo3_brain_age_256/version_0/hparams.yaml"
    
    if os.path.exists(default_path):
        return default_path
    
    # Search for alternative configs
    runs_dir = "runs/Task003_FOMO3/unet_xl"
    if os.path.exists(runs_dir):
        for experiment_dir in os.listdir(runs_dir):
            config_path = os.path.join(runs_dir, experiment_dir, "version_0", "hparams.yaml")
            if os.path.exists(config_path):
                logging.warning(f"Using alternative config: {config_path}")
                return config_path
    
    raise FileNotFoundError(
        f"No Task3 config found. Please train a model first or specify --config_path"
    )


def validate_checkpoint_compatibility(checkpoint_path: str) -> bool:
    """
    Validate that checkpoint is compatible with Task3 regression.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        model_info = ModelLoader.detect_model_architecture(checkpoint_path)
        
        # Check task type
        task_type = model_info.get('config', {}).get('task_type', 'unknown')
        if task_type != 'regression':
            logging.error(f"Checkpoint is for {task_type}, expected regression")
            return False
        
        # Check modalities
        modalities = model_info.get('modalities', [])
        if len(modalities) != 2:
            logging.error(f"Checkpoint has {len(modalities)} modalities, expected 2 for Task3")
            return False
            
        if not set(modalities).issubset({'T1', 'T2'}):
            logging.error(f"Checkpoint modalities {modalities} don't match Task3 (T1, T2)")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating checkpoint compatibility: {e}")
        return False


def predict_brain_age(
    t1_path: str,
    t2_path: str,
    checkpoint_path: str,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    tta_enabled: bool = False
) -> float:
    """
    Predict brain age from T1 and T2 modalities.
    
    Args:
        t1_path: Path to T1 NIfTI file
        t2_path: Path to T2 NIfTI file  
        checkpoint_path: Path to trained model checkpoint
        config: Model configuration dictionary
        device: PyTorch device for inference
        tta_enabled: Whether to enable test-time augmentation
        
    Returns:
        Predicted brain age in years
    """
    if device is None:
        device = select_device('auto')
    
    # Prepare modality paths (T1 first, T2 second for Task3)
    modality_paths = [t1_path, t2_path]
    
    # Create prediction config
    predict_config = {
        "task_type": "regression", 
        "crop_to_nonzero": True,
        "norm_op": "zscore",  # Standard normalization
        "num_classes": 1,  # Regression output
        "keep_aspect_ratio": True,
        "patch_size": config.get("patch_size", [256, 256, 32]),
        "model_path": checkpoint_path
    }
    
    # Run inference
    predictions, _ = predict_from_config(
        modality_paths=modality_paths,
        predict_config=predict_config,
        checkpoint_path=checkpoint_path
    )
    
    # Extract scalar prediction
    if isinstance(predictions, torch.Tensor):
        if predictions.dim() > 0:
            age_prediction = predictions.item() if predictions.numel() == 1 else predictions[0].item()
        else:
            age_prediction = predictions.item()
    else:
        age_prediction = float(predictions)
    
    # Denormalize if age normalization was used during training
    if config.get("age_normalization", False):
        age_mean = config.get("age_mean", 61.87)
        age_std = config.get("age_std", 15.09)
        age_prediction = denormalize_age(age_prediction, age_mean, age_std)
    
    return age_prediction


def save_prediction_result(predicted_age: float, output_path: str) -> None:
    """
    Save prediction result to text file.
    
    Args:
        predicted_age: Predicted age in years
        output_path: Output file path
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Format age to reasonable precision (1 decimal place)
    formatted_age = f"{predicted_age:.1f}"
    
    # Save as plain text
    output_file = output_path if output_path.endswith('.txt') else f"{output_path}.txt"
    with open(output_file, 'w') as f:
        f.write(formatted_age)
    
    logging.info(f"Prediction saved to: {output_file}")


def validate_args(args) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments
        
    Raises:
        ValueError: If arguments are invalid
        FileNotFoundError: If required files don't exist
    """
    # Validate modalities
    validate_modality_files(args.modalities)
    
    # Validate checkpoint if specified
    if args.checkpoint_path and not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    # Validate config if specified
    if args.config_path and not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config not found: {args.config_path}")


def main() -> int:
    """
    Main inference function.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="FOMO Task3 Brain Age Regression Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto-detected checkpoint
  python predict_task3.py --modalities t1.nii.gz t2.nii.gz --output_dir ./results
  
  # Advanced usage with custom settings
  python predict_task3.py --modalities t1.nii.gz t2.nii.gz --output_dir ./results \\
    --checkpoint_path custom.ckpt --device cuda:1 --tta
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--modalities", 
        nargs=2, 
        required=True,
        metavar=('T1_PATH', 'T2_PATH'),
        help="Paths to T1 and T2 NIfTI files (in that order)"
    )
    parser.add_argument(
        "--output_dir", 
        required=True,
        help="Directory to save prediction results"
    )
    
    # Optional arguments
    parser.add_argument(
        "--checkpoint_path", 
        default=None,
        help="Path to model checkpoint (auto-detect if not specified)"
    )
    parser.add_argument(
        "--config_path", 
        default=None,
        help="Path to model config file (auto-detect if not specified)"
    )
    parser.add_argument(
        "--device", 
        default="auto",
        help="Device for inference (auto, cpu, cuda, cuda:X)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Inference batch size"
    )
    parser.add_argument(
        "--tta", 
        action="store_true",
        help="Enable test-time augmentation for improved accuracy"
    )
    parser.add_argument(
        "--output_name", 
        default="brain_age_prediction",
        help="Base name for output file (without extension)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    
    try:
        # Validate arguments
        validate_args(args)
        
        # Find checkpoint and config if not specified
        checkpoint_path = args.checkpoint_path or find_default_checkpoint()
        config_path = args.config_path or find_default_config()
        
        logging.info(f"Using checkpoint: {checkpoint_path}")
        logging.info(f"Using config: {config_path}")
        
        # Validate checkpoint compatibility
        if not validate_checkpoint_compatibility(checkpoint_path):
            raise ValueError("Checkpoint is not compatible with Task3 brain age regression")
        
        # Load configuration
        config = load_task3_config(config_path)
        
        # Select device
        device = select_device(args.device)
        
        # Run prediction
        logging.info("Running brain age prediction...")
        predicted_age = predict_brain_age(
            t1_path=args.modalities[0],
            t2_path=args.modalities[1], 
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
            tta_enabled=args.tta
        )
        
        # Save result
        output_path = os.path.join(args.output_dir, f"{args.output_name}.txt")
        save_prediction_result(predicted_age, output_path)
        
        # Print result
        print(f"🧠 Brain Age Prediction: {predicted_age:.1f} years")
        print(f"📄 Result saved to: {output_path}")
        
        return 0
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logging.error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
