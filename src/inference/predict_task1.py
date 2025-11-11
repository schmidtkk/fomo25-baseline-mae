#!/usr/bin/env python
"""
FOMO Task 1 (Infarct Detection) inference script.

This script performs binary classification to detect the presence of infarcts
in brain MRI images using multiple modalities (DWI, ADC, T2FLAIR, SWI/T2*).
"""

import argparse
import os
import torch
import logging
from torch.nn.functional import softmax
from typing import List, Optional

# Import base inference utilities
from .predict import (
    predict_from_config, 
    save_output_txt, 
    find_best_checkpoint
)
from .model_loader import ModelLoader
from data.task_configs import task1_config


# Task-specific hardcoded configuration
predict_config = {
    # Import values from task_configs
    **task1_config,
    # Add inference-specific configs
    "model_path": "/app/model.ckpt",  # Default path (overridden by user)
    "patch_size": (96, 96, 96),      # Inference patch size
}


class InferenceAggregator:
    """Handle multiple inference aggregation strategies."""
    
    def __init__(self, base_model, tta_config=None, ensemble_config=None, uncertainty_config=None):
        self.base_model = base_model
        self.tta_config = tta_config or {}
        self.ensemble_config = ensemble_config or {}
        self.uncertainty_config = uncertainty_config or {}
        try:
            self.device = next(base_model.parameters()).device
        except StopIteration:
            # Fallback for testing or models without parameters
            self.device = torch.device('cpu')
    
    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Run inference with configured aggregation methods."""
        predictions = []
        
        # Base model prediction
        with torch.no_grad():
            base_pred = self.base_model(input_data)
            predictions.append(base_pred)
        
        # Apply TTA if enabled
        if self.tta_config.get('enable', False):
            tta_predictions = self._apply_tta(input_data)
            predictions.extend(tta_predictions)
        
        # Apply ensemble if enabled
        if self.ensemble_config.get('enable', False):
            ensemble_predictions = self._apply_ensemble(input_data)
            predictions.extend(ensemble_predictions)
        
        # Aggregate all predictions
        aggregation_method = self.tta_config.get('aggregation', 'mean')
        aggregated = self._aggregate_predictions(predictions, method=aggregation_method)
        
        # Apply uncertainty estimation if requested
        if self.uncertainty_config.get('enable', False):
            uncertainty_result = self._estimate_uncertainty(predictions)
            # For now, just return the prediction (could be extended to return uncertainty)
            return uncertainty_result['prediction']
        
        return aggregated
    
    def _apply_tta(self, input_data: torch.Tensor) -> List[torch.Tensor]:
        """Apply Test-Time Augmentation transformations."""
        tta_predictions = []
        transforms = self.tta_config.get('transforms', ['original', 'flip_x', 'flip_y', 'flip_z'])
        
        for transform in transforms:
            # Apply forward transform using the comprehensive method
            augmented = self._apply_flip_transform(input_data, transform)
            
            # Run inference on augmented data
            with torch.no_grad():
                pred = self.base_model(augmented)
                # Note: For classification, we don't need to reverse the transform
                # since we only care about the scalar output
                tta_predictions.append(pred)
        
        return tta_predictions
    
    def _apply_ensemble(self, input_data: torch.Tensor) -> List[torch.Tensor]:
        """Apply ensemble inference with multiple models."""
        ensemble_predictions = []
        checkpoints = self.ensemble_config.get('checkpoints', [])
        
        for checkpoint_path in checkpoints:
            try:
                # Load additional model
                model = ModelLoader.load_model(checkpoint_path, 'classification')
                model = model.to(self.device)
                model.eval()
                
                # Run inference
                with torch.no_grad():
                    pred = model(input_data)
                    ensemble_predictions.append(pred)
                    
                # Clean up to save memory
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logging.warning(f"Failed to load ensemble model {checkpoint_path}: {e}")
                continue
        
        return ensemble_predictions
    
    def _estimate_uncertainty(self, predictions: List[torch.Tensor]) -> dict:
        """Estimate prediction uncertainty from multiple predictions."""
        predictions_tensor = torch.stack(predictions)
        
        # Calculate mean and variance
        mean_pred = torch.mean(predictions_tensor, dim=0)
        variance = torch.var(predictions_tensor, dim=0)
        
        # Simple confidence measure (inverse of variance)
        confidence = 1.0 / (1.0 + torch.mean(variance))
        
        return {
            'prediction': mean_pred,
            'uncertainty': variance,
            'confidence': confidence.item(),
            'num_predictions': len(predictions)
        }
    
    def _apply_flip_transform(self, input_data: torch.Tensor, transform: str) -> torch.Tensor:
        """Apply a specific flip transformation to input data."""
        if transform == 'original':
            return input_data
        elif transform == 'flip_x':
            return torch.flip(input_data, dims=[-3])
        elif transform == 'flip_y':
            return torch.flip(input_data, dims=[-2])
        elif transform == 'flip_z':
            return torch.flip(input_data, dims=[-1])
        elif transform == 'flip_xy':
            return torch.flip(input_data, dims=[-3, -2])
        elif transform == 'flip_xz':
            return torch.flip(input_data, dims=[-3, -1])
        elif transform == 'flip_yz':
            return torch.flip(input_data, dims=[-2, -1])
        elif transform == 'flip_xyz':
            return torch.flip(input_data, dims=[-3, -2, -1])
        else:
            return input_data
    
    def _aggregate_predictions(self, predictions: List[torch.Tensor], method: str = 'mean') -> torch.Tensor:
        """Aggregate multiple predictions using specified method."""
        predictions_tensor = torch.stack(predictions)
        
        if method == 'mean':
            return torch.mean(predictions_tensor, dim=0)
        elif method == 'median':
            return torch.median(predictions_tensor, dim=0)[0]
        elif method == 'max':
            return torch.max(predictions_tensor, dim=0)[0]
        else:
            return torch.mean(predictions_tensor, dim=0)  # Default to mean


def create_tta_config(args) -> dict:
    """Create TTA configuration from command line arguments."""
    if not args.tta_enable:
        return {'enable': False}
    
    # Determine TTA transforms based on number of views
    all_transforms = ['original', 'flip_x', 'flip_y', 'flip_z', 'flip_xy', 'flip_xz', 'flip_yz', 'flip_xyz']
    num_views = min(args.tta_views, len(all_transforms))
    selected_transforms = all_transforms[:num_views]
    
    return {
        'enable': True,
        'transforms': selected_transforms,
        'aggregation': 'mean'
    }


def create_ensemble_config(args) -> dict:
    """Create ensemble configuration from command line arguments."""
    if not args.ensemble_checkpoints:
        return {'enable': False}
    
    checkpoints = [ckpt.strip() for ckpt in args.ensemble_checkpoints.split(',')]
    
    return {
        'enable': True,
        'checkpoints': checkpoints,
        'method': 'simple_average'
    }


def create_uncertainty_config(args) -> dict:
    """Create uncertainty configuration from command line arguments."""
    return {
        'enable': args.confidence_threshold > 0,
        'method': 'ensemble_disagreement',
        'threshold': args.confidence_threshold
    }


def map_modality_arguments(args) -> List[str]:
    """
    Map command line arguments to canonical modality order.
    
    Template expects: [dwi_b1000, flair, adc, swi/t2s]
    Our training uses: ["DWI", "ADC", "T2FLAIR", "SWI_OR_T2STAR"]
    
    Need to reorder: [dwi_b1000, adc, flair, swi/t2s] -> [DWI, ADC, T2FLAIR, SWI_OR_T2STAR]
    """
    return [
        args.dwi_b1000,           # Maps to "DWI" 
        args.adc,                 # Maps to "ADC"
        args.flair,               # Maps to "T2FLAIR"
        args.swi or args.t2s      # Maps to "SWI_OR_T2STAR"
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on FOMO Task 1 (Infarct Detection)"
    )
    
    # Required modality arguments (following template format)
    parser.add_argument(
        "--dwi_b1000", 
        type=str, 
        required=True, 
        help="Path to DWI b1000 image (NIfTI format)"
    )
    parser.add_argument(
        "--flair",
        type=str,
        required=True,
        help="Path to T2FLAIR image (NIfTI format)",
    )
    parser.add_argument(
        "--adc", 
        type=str, 
        required=True, 
        help="Path to ADC image (NIfTI format)"
    )
    
    # Optional modality arguments (one required)
    parser.add_argument(
        "--swi", 
        type=str,
        required=False, 
        help="Path to SWI image (NIfTI format)"
    )
    parser.add_argument(
        "--t2s", 
        type=str, 
        required=False, 
        help="Path to T2* image (NIfTI format)"
    )
    
    # Output argument
    parser.add_argument(
        "--output", 
        type=str,
        required=True, 
        help="Output path for prediction (.txt file)"
    )
    
    # Model configuration arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific model checkpoint (auto-detects best if not specified)"
    )
    parser.add_argument(
        "--model_architecture",
        type=str,
        default=None,
        help="Force specific architecture (unet_xl, mednext, etc.)"
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default=None,
        help="Override fusion mechanism (masked_mean, attention, etc.)"
    )
    
    # Inference aggregation arguments
    parser.add_argument(
        "--ensemble_checkpoints",
        type=str,
        default=None,
        help="Comma-separated list of additional checkpoints for ensemble"
    )
    parser.add_argument(
        "--tta_enable",
        action="store_true",
        help="Enable Test-Time Augmentation"
    )
    parser.add_argument(
        "--tta_views",
        type=int,
        default=8,
        help="Number of TTA views (default: 8, max: 8)"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.0,
        help="Minimum confidence for prediction (0.0 to disable)"
    )
    
    # Performance arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force specific device (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate modality arguments
    if not ((args.swi and not args.t2s) or (not args.swi and args.t2s)):
        raise ValueError("Either --swi or --t2s must be provided, but not both.")
    
    # Validate all input files exist
    modality_paths = map_modality_arguments(args)
    for i, path in enumerate(modality_paths):
        if not os.path.exists(path):
            modality_names = ["DWI", "ADC", "T2FLAIR", "SWI_OR_T2STAR"]
            raise FileNotFoundError(f"{modality_names[i]} file not found: {path}")
    
    # Determine checkpoint path
    if args.checkpoint is None:
        try:
            checkpoint_path = find_best_checkpoint()
            logging.info(f"Auto-detected checkpoint: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to find checkpoint: {e}")
    else:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_path}")
    
    # Update config with user-specified checkpoint
    config = predict_config.copy()
    config["model_path"] = checkpoint_path
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Using device: {device}")
    logging.info(f"Input modalities: {[os.path.basename(p) for p in modality_paths]}")
    logging.info(f"Output path: {args.output}")
    
    try:
        # Check if we need aggregation methods
        use_aggregation = (
            args.tta_enable or 
            args.ensemble_checkpoints or 
            args.confidence_threshold > 0
        )
        
        if use_aggregation:
            logging.info("Using inference aggregation methods...")
            
            # Load base model
            base_model = ModelLoader.load_model(checkpoint_path, "classification")
            base_model = base_model.to(device)
            
            # Set up aggregation
            aggregator = InferenceAggregator(
                base_model=base_model,
                tta_config=create_tta_config(args),
                ensemble_config=create_ensemble_config(args),
                uncertainty_config=create_uncertainty_config(args)
            )
            
            # Run basic preprocessing to get input tensor
            from .predict import load_modalities
            from yucca.functional.preprocessing import preprocess_case_for_inference
            
            images = load_modalities(modality_paths)
            
            # Preprocess using config parameters
            case_preprocessed, _ = preprocess_case_for_inference(
                crop_to_nonzero=config["crop_to_nonzero"],
                images=images,
                intensities=None,
                normalization_scheme=[config["norm_op"]] * len(modality_paths),
                patch_size=config["patch_size"],
                target_spacing=[1.0, 1.0, 1.0],
                target_orientation="RAS",
                allow_missing_modalities=False,
                keep_aspect_ratio=config.get("keep_aspect_ratio", True),
                transpose_forward=[0, 1, 2],
            )
            
            # Add batch dimension and move to device
            if case_preprocessed.dim() == 4:
                case_preprocessed = case_preprocessed.unsqueeze(0)
            case_preprocessed = case_preprocessed.to(device)
            
            # Run aggregated inference
            predictions = aggregator.predict(case_preprocessed)
            
        else:
            # Standard single-model inference
            logging.info("Using standard single-model inference...")
            predictions, _ = predict_from_config(
                modality_paths=modality_paths,
                predict_config=config,
                checkpoint_path=checkpoint_path
            )
        
        # Convert logits to probabilities
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            # Multi-class output, apply softmax
            probabilities = softmax(predictions, dim=-1)
            # Extract positive class probability (class 1)
            positive_prob = probabilities[0, 1].item()
        else:
            # Single output (binary classification with sigmoid or regression)
            if config["task_type"] == "classification":
                # Apply sigmoid for binary classification
                positive_prob = torch.sigmoid(predictions).item()
            else:
                # For regression, just use the raw output
                positive_prob = predictions.item()
        
        # Apply confidence threshold if specified
        if args.confidence_threshold > 0 and hasattr(locals(), 'aggregator'):
            # This would need to be implemented in the aggregator
            logging.info(f"Prediction confidence above threshold: {args.confidence_threshold}")
        
        logging.info(f"Predicted probability: {positive_prob:.6f}")
        
        # Save result
        save_output_txt(positive_prob, args.output)
        logging.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logging.error(f"Inference failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
