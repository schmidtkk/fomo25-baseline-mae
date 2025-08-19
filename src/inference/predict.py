#!/usr/bin/env python
"""
Base inference framework for FOMO25 challenge.

This module provides shared utilities for all task-specific inference scripts,
including model loading, preprocessing, and aggregation methods.
"""

import os
import torch
import numpy as np
import nibabel as nib
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path

# Import our models and utilities
from models.supervised_base import BaseSupervisedModel
from data.task_configs import task1_config, task2_config, task3_config
from yucca.functional.preprocessing import preprocess_case_for_inference


def get_task_config(taskid: int) -> Dict[str, Any]:
    """Get task configuration based on task ID."""
    if taskid == 1:
        return task1_config
    elif taskid == 2:
        return task2_config
    elif taskid == 3:
        return task3_config
    else:
        raise ValueError(f"Unknown taskid: {taskid}. Supported IDs are 1, 2, and 3")


class ModelLoader:
    """Flexible model loader supporting multiple architectures and checkpoints."""
    
    @staticmethod
    def detect_model_architecture(checkpoint_path: str) -> Dict[str, Any]:
        """
        Analyze checkpoint to determine model architecture and configuration.
        
        Args:
            checkpoint_path: Path to PyTorch Lightning checkpoint
            
        Returns:
            Dictionary containing model architecture information
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            keys = set(state_dict.keys())
            
            # Detect encoder type
            if any('model.encoder.encoders.' in key for key in keys):
                encoder_type = "multi_encoder"
                modalities = ModelLoader._extract_modality_names(keys)
                fusion_type = ModelLoader._detect_fusion_type(keys)
            else:
                encoder_type = "single_encoder"
                modalities = []
                fusion_type = None
            
            # Detect network architecture from layer patterns
            architecture = ModelLoader._detect_network_architecture(keys)
            
            # Extract configuration from hyperparameters if available
            config = checkpoint.get('hyper_parameters', {}).get('config', {})
            
            return {
                'architecture': architecture,
                'encoder_type': encoder_type,
                'modalities': modalities,
                'fusion_type': fusion_type,
                'config': config,
                'checkpoint_path': checkpoint_path
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze checkpoint {checkpoint_path}: {str(e)}")
    
    @staticmethod
    def _extract_modality_names(keys: set) -> List[str]:
        """Extract modality names from checkpoint keys."""
        modalities = set()
        for key in keys:
            if 'model.encoder.encoders.' in key:
                # Extract modality name: model.encoder.encoders.{MODALITY}.layer...
                parts = key.split('.')
                if len(parts) >= 4:
                    modality = parts[3]
                    modalities.add(modality)
        return sorted(list(modalities))
    
    @staticmethod
    def _detect_fusion_type(keys: set) -> Optional[str]:
        """Detect fusion mechanism from checkpoint keys."""
        if any('fusions' in key for key in keys):
            if any('attention_fc' in key for key in keys):
                return "attention"
            elif any('cross_modality_attention' in key for key in keys):
                return "attention"
            elif any('modality_weights' in key for key in keys):
                return "learnable_weighted"
            elif any('channel_gate' in key for key in keys):
                return "channel_gated"
            else:
                return "masked_mean"  # Default fusion type
        return None
    
    @staticmethod
    def _detect_network_architecture(keys: set) -> str:
        """Detect network architecture from layer patterns."""
        # Look for architecture-specific patterns
        if any('mednext' in key.lower() for key in keys):
            return "mednext"
        elif any('decoder' in key and 'encoder' in key for key in keys):
            # UNet-style architecture
            if any('1024' in str(key) for key in keys):
                return "unet_xl"
            else:
                return "unet_b"
        else:
            return "unet_xl"  # Default fallback
    
    @staticmethod
    def load_model(checkpoint_path: str, task_type: str = "classification") -> torch.nn.Module:
        """
        Load model from checkpoint with automatic architecture detection.
        
        Args:
            checkpoint_path: Path to checkpoint file
            task_type: Task type (classification, regression, segmentation)
            
        Returns:
            Loaded PyTorch model ready for inference
        """
        # Analyze checkpoint
        model_info = ModelLoader.detect_model_architecture(checkpoint_path)
        
        # Reconstruct configuration
        config = ModelLoader._reconstruct_config(model_info, task_type)
        
        # Create model using our factory method
        model = BaseSupervisedModel.create(
            task_type=task_type,
            config=config
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Handle potential compilation artifacts
        if any('_orig_mod' in key for key in state_dict.keys()):
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                cleaned_state_dict[new_key] = value
            state_dict = cleaned_state_dict
        
        # Load state dict with error handling
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logging.warning(f"Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
        
        model.eval()
        return model
    
    @staticmethod
    def _reconstruct_config(model_info: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Reconstruct configuration dictionary for model creation."""
        # Start with saved config if available
        config = model_info.get('config', {}).copy()
        
        # Set essential parameters
        config.update({
            'model_name': model_info['architecture'],
            'task_type': task_type,
            'use_multi_encoder': model_info['encoder_type'] == 'multi_encoder',
        })
        
        # Add multi-encoder specific config
        if model_info['encoder_type'] == 'multi_encoder':
            config.update({
                'multi_encoder_modalities': model_info['modalities'],
                'fusion_type': model_info.get('fusion_type', 'masked_mean'),
                'global_vocab': ["t1", "t2", "flair", "dwi", "other"],
            })
            
            # Default modality mapping for FOMO tasks
            if not config.get('modality_to_global_group'):
                config['modality_to_global_group'] = {
                    "DWI": "dwi",
                    "ADC": "dwi", 
                    "T2FLAIR": "flair",
                    "SWI_OR_T2STAR": "other",
                    "T1": "t1",
                    "T2": "t2"
                }
        
        # Set reasonable defaults for missing parameters
        config.setdefault('num_classes', 2 if task_type == 'classification' else 1)
        config.setdefault('num_modalities', len(model_info.get('modalities', [4])))
        config.setdefault('patch_size', (96, 96, 96))
        config.setdefault('starting_filters', 64)
        config.setdefault('model_dimensions', '3D')
        config.setdefault('deep_supervision', False)
        
        return config


def find_best_checkpoint(runs_dir: str = "runs", task_name: str = "Task001_FOMO1") -> str:
    """
    Find the best available checkpoint for a given task.
    
    Args:
        runs_dir: Base directory containing training runs
        task_name: Task directory name (Task001_FOMO1, etc.)
        
    Returns:
        Path to best checkpoint
    """
    task_dir = os.path.join(runs_dir, task_name)
    
    if not os.path.exists(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    
    # Look for checkpoint directories
    checkpoint_candidates = []
    
    for root, dirs, files in os.walk(task_dir):
        if "checkpoints" in dirs:
            checkpoint_dir = os.path.join(root, "checkpoints")
            best_ckpt = os.path.join(checkpoint_dir, "best.ckpt")
            
            if os.path.exists(best_ckpt):
                # Score based on directory name (prefer non-ablation models)
                score = 0
                if "ablation" not in root:
                    score += 10
                if "optimized" in root or "baseline" in root:
                    score += 5
                
                checkpoint_candidates.append((score, best_ckpt))
    
    if not checkpoint_candidates:
        raise FileNotFoundError(f"No checkpoints found in {task_dir}")
    
    # Return highest scoring checkpoint
    checkpoint_candidates.sort(reverse=True)
    return checkpoint_candidates[0][1]


def load_modalities(modality_paths: List[str]) -> List[nib.Nifti1Image]:
    """Load modality images from provided paths."""
    images = []
    
    for path in modality_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modality file not found: {path}")
        
        try:
            img = nib.load(path)
            images.append(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {path}: {str(e)}")
    
    return images


def save_output_txt(number: Union[float, int], output_path: str) -> None:
    """Save a number (float or int) as plain text to a file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if not output_path.endswith(".txt"):
        output_path = output_path + ".txt"
    
    with open(output_path, "w") as f:
        f.write(f"{number}")


def predict_from_config(
    modality_paths: List[str],
    predict_config: Dict[str, Any],
    checkpoint_path: Optional[str] = None
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """
    Run inference on input modality images using task-specific configuration.
    
    Args:
        modality_paths: Paths to input modality images
        predict_config: Dictionary containing prediction configuration
        checkpoint_path: Optional specific checkpoint path
        
    Returns:
        Tuple of (predictions, reference_affine)
    """
    # Load input images
    images = load_modalities(modality_paths)
    
    # Extract configuration
    task_type = predict_config["task_type"]
    crop_to_nonzero = predict_config["crop_to_nonzero"]
    norm_op = predict_config["norm_op"]
    num_classes = predict_config["num_classes"]
    keep_aspect_ratio = predict_config.get("keep_aspect_ratio", True)
    patch_size = predict_config["patch_size"]
    
    # Use specified checkpoint or find best one
    if checkpoint_path is None:
        checkpoint_path = predict_config["model_path"]
    
    # Define preprocessing parameters
    normalization_scheme = [norm_op] * len(modality_paths)
    target_spacing = [1.0, 1.0, 1.0]  # Isotropic 1mm spacing
    target_orientation = "RAS"
    
    # Apply preprocessing to match training pipeline
    case_preprocessed, case_properties = preprocess_case_for_inference(
        crop_to_nonzero=crop_to_nonzero,
        images=images,
        intensities=None,
        normalization_scheme=normalization_scheme,
        patch_size=patch_size,
        target_size=None,
        target_spacing=target_spacing,
        target_orientation=target_orientation,
        allow_missing_modalities=False,
        keep_aspect_ratio=keep_aspect_ratio,
        transpose_forward=[0, 1, 2],
    )
    
    # Load model using our flexible loader
    model = ModelLoader.load_model(checkpoint_path, task_type)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    case_preprocessed = case_preprocessed.to(device)
    
    # Add batch dimension if needed
    if case_preprocessed.dim() == 4:  # [C, D, H, W] -> [1, C, D, H, W]
        case_preprocessed = case_preprocessed.unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        if task_type == "segmentation":
            # For segmentation, we might need sliding window
            predictions = model.model.predict(
                data=case_preprocessed,
                mode="3D",
                mirror=False,
                overlap=0.5,
                patch_size=patch_size,
                sliding_window_prediction=True,
            )
        else:
            # For classification/regression, direct forward pass
            predictions = model(case_preprocessed)
    
    # Return predictions and reference affine for potential reverse preprocessing
    return predictions, images[0].affine if len(images) > 0 else None


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # This is a utility module, main functionality in task-specific scripts
    import logging
    logging.info("FOMO25 Inference Framework")
    logging.info("Use task-specific scripts like predict_task1.py for inference")
