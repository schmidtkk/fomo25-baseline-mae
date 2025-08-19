#!/usr/bin/env python
"""
Model loading utilities for FOMO25 inference pipeline.

This module provides flexible model loading capabilities supporting:
- Automatic architecture detection from checkpoints
- Multi-encoder and single-encoder models
- Different fusion mechanisms
- Configuration reconstruction
"""

import os
import torch
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from models.supervised_base import BaseSupervisedModel


class ModelLoader:
    """Enhanced model loader with comprehensive architecture support."""
    
    # Architecture patterns for detection
    ARCHITECTURE_PATTERNS = {
        'unet_xl': ['decoder', 'encoder', '1024'],
        'unet_b': ['decoder', 'encoder', '512'],
        'unet_l': ['decoder', 'encoder', '768'],
        'mednext': ['mednext', 'convnext'],
        'mednext_b': ['mednext', 'base'],
        'mednext_l': ['mednext', 'large']
    }
    
    # Fusion type patterns
    FUSION_PATTERNS = {
        'attention': ['attention_fc', 'cross_modality_attention'],
        'channel_attention': ['channel_attention'],
        'spatial_attention': ['spatial_attention'],
        'hybrid_attention': ['hybrid_attention'],
        'learnable_weighted': ['modality_weights'],
        'channel_gated': ['channel_gate'],
        'uncertainty_weighted': ['uncertainty'],
        'masked_mean': ['gamma', 'post_scale']  # Default fusion
    }
    
    @staticmethod
    def detect_model_architecture(checkpoint_path: str) -> Dict[str, Any]:
        """
        Comprehensive checkpoint analysis for architecture detection.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            keys = set(state_dict.keys())
            
            # Extract basic information
            encoder_info = ModelLoader._analyze_encoder_structure(keys)
            fusion_info = ModelLoader._analyze_fusion_structure(keys)
            architecture_info = ModelLoader._analyze_network_architecture(keys)
            
            # Get hyperparameters and config if available
            hyperparams = checkpoint.get('hyper_parameters', {})
            saved_config = hyperparams.get('config', {})
            
            result = {
                'checkpoint_path': checkpoint_path,
                'encoder_type': encoder_info['type'],
                'modalities': encoder_info['modalities'],
                'fusion_type': fusion_info['type'],
                'fusion_details': fusion_info['details'],
                'architecture': architecture_info['name'],
                'architecture_details': architecture_info['details'],
                'saved_config': saved_config,
                'hyperparameters': hyperparams,
                'state_dict_keys': keys
            }
            
            logging.info(f"Detected architecture: {result['architecture']}")
            logging.info(f"Encoder type: {result['encoder_type']}")
            if result['modalities']:
                logging.info(f"Modalities: {result['modalities']}")
            if result['fusion_type']:
                logging.info(f"Fusion type: {result['fusion_type']}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze checkpoint {checkpoint_path}: {str(e)}")
    
    @staticmethod
    def _analyze_encoder_structure(keys: set) -> Dict[str, Any]:
        """Analyze encoder structure from state dict keys."""
        # Check for multi-encoder structure
        multi_encoder_keys = [key for key in keys if 'model.encoder.encoders.' in key]
        
        if multi_encoder_keys:
            # Extract modality names
            modalities = set()
            for key in multi_encoder_keys:
                parts = key.split('.')
                if len(parts) >= 4 and parts[0] == 'model' and parts[1] == 'encoder' and parts[2] == 'encoders':
                    modality = parts[3]
                    modalities.add(modality)
            
            return {
                'type': 'multi_encoder',
                'modalities': sorted(list(modalities)),
                'num_encoders': len(modalities)
            }
        else:
            return {
                'type': 'single_encoder',
                'modalities': [],
                'num_encoders': 1
            }
    
    @staticmethod
    def _analyze_fusion_structure(keys: set) -> Dict[str, Any]:
        """Analyze fusion mechanism from state dict keys."""
        fusion_keys = [key for key in keys if 'model.encoder.fusions.' in key]
        
        if not fusion_keys:
            return {'type': None, 'details': {}}
        
        # Detect fusion type based on key patterns
        detected_type = 'masked_mean'  # Default
        details = {}
        
        for fusion_type, patterns in ModelLoader.FUSION_PATTERNS.items():
            if any(pattern in key for key in fusion_keys for pattern in patterns):
                detected_type = fusion_type
                break
        
        # Extract fusion details
        details['num_scales'] = len(set(
            key.split('.')[3] for key in fusion_keys 
            if len(key.split('.')) > 3 and key.split('.')[3].isdigit()
        ))
        
        details['has_gamma'] = any('gamma' in key for key in fusion_keys)
        details['has_attention'] = any('attention' in key for key in fusion_keys)
        details['has_weights'] = any('weights' in key for key in fusion_keys)
        
        return {
            'type': detected_type,
            'details': details
        }
    
    @staticmethod
    def _analyze_network_architecture(keys: set) -> Dict[str, Any]:
        """Detect network architecture from layer patterns."""
        key_str = ' '.join(keys)
        
        # Try to match known architecture patterns
        for arch_name, patterns in ModelLoader.ARCHITECTURE_PATTERNS.items():
            if all(pattern.lower() in key_str.lower() for pattern in patterns):
                return {
                    'name': arch_name,
                    'details': {
                        'patterns_matched': patterns,
                        'confidence': 'high'
                    }
                }
        
        # Fallback detection based on common patterns
        if 'decoder' in key_str and 'encoder' in key_str:
            # UNet-style architecture
            if '1024' in key_str:
                arch_name = 'unet_xl'
            elif '512' in key_str:
                arch_name = 'unet_b'
            else:
                arch_name = 'unet_b'
        elif 'mednext' in key_str.lower():
            arch_name = 'mednext'
        else:
            arch_name = 'unet_xl'  # Conservative default
        
        return {
            'name': arch_name,
            'details': {
                'patterns_matched': [],
                'confidence': 'medium',
                'fallback_reason': 'pattern_based_guess'
            }
        }
    
    @staticmethod
    def reconstruct_config(model_info: Dict[str, Any], task_type: str = "classification") -> Dict[str, Any]:
        """
        Reconstruct complete configuration for model creation.
        """
        # Start with saved configuration if available
        config = model_info.get('saved_config', {}).copy()
        
        # Override/set essential parameters
        config.update({
            'model_name': model_info['architecture'],
            'task_type': task_type,
            'use_multi_encoder': model_info['encoder_type'] == 'multi_encoder',
        })
        
        # Multi-encoder specific configuration
        if model_info['encoder_type'] == 'multi_encoder':
            config.update({
                'multi_encoder_modalities': model_info['modalities'],
                'fusion_type': model_info['fusion_type'] or 'masked_mean',
                'global_vocab': ["t1", "t2", "flair", "dwi", "other"],
            })
            
            # Set up modality mapping
            config.setdefault('modality_to_global_group', 
                ModelLoader._create_default_modality_mapping(model_info['modalities'])
            )
        
        # Set reasonable defaults for missing critical parameters
        config.setdefault('num_classes', 2 if task_type == 'classification' else 1)
        config.setdefault('num_modalities', len(model_info.get('modalities', [4])))
        config.setdefault('patch_size', (96, 96, 96))
        config.setdefault('starting_filters', 64)
        config.setdefault('model_dimensions', '3D')
        config.setdefault('deep_supervision', False)
        config.setdefault('precision', '32-true')
        config.setdefault('version_dir', '/tmp')  # Temporary directory for inference
        
        # Architecture-specific defaults
        if 'xl' in model_info['architecture']:
            config.setdefault('starting_filters', 64)
        elif 'mednext' in model_info['architecture']:
            config.setdefault('starting_filters', 32)
        
        return config
    
    @staticmethod
    def _create_default_modality_mapping(modalities: List[str]) -> Dict[str, str]:
        """Create default modality to global group mapping."""
        mapping = {}
        
        for modality in modalities:
            mod_lower = modality.lower()
            
            # Mapping logic based on common modality names
            if 'dwi' in mod_lower or 'adc' in mod_lower:
                mapping[modality] = 'dwi'
            elif 'flair' in mod_lower:
                mapping[modality] = 'flair'
            elif 't1' in mod_lower and 't1' == mod_lower:
                mapping[modality] = 't1'
            elif 't2' in mod_lower and 't2' == mod_lower:
                mapping[modality] = 't2'
            elif any(token in mod_lower for token in ['swi', 't2s', 't2star', 't2*']):
                mapping[modality] = 'other'
            else:
                mapping[modality] = 'other'  # Default fallback
        
        return mapping
    
    @staticmethod
    def load_model(checkpoint_path: str, task_type: str = "classification", device: str = None) -> torch.nn.Module:
        """
        Load model with comprehensive error handling and optimization.
        """
        # Detect architecture
        model_info = ModelLoader.detect_model_architecture(checkpoint_path)
        
        # Reconstruct configuration
        config = ModelLoader.reconstruct_config(model_info, task_type)
        
        try:
            # Create model using factory method
            model = BaseSupervisedModel.create(
                task_type=task_type,
                config=config
            )
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            # Handle potential compilation artifacts
            state_dict = ModelLoader._clean_state_dict(state_dict)
            
            # Load weights with detailed error reporting
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            # Report loading results
            if missing_keys:
                logging.warning(f"Missing keys when loading checkpoint: {len(missing_keys)} keys")
                if len(missing_keys) < 10:  # Don't spam if too many
                    logging.debug(f"Missing keys: {missing_keys}")
            
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading checkpoint: {len(unexpected_keys)} keys")
                if len(unexpected_keys) < 10:
                    logging.debug(f"Unexpected keys: {unexpected_keys}")
            
            # Set to evaluation mode
            model.eval()
            
            # Move to specified device
            if device:
                model = model.to(device)
            elif torch.cuda.is_available():
                model = model.to('cuda')
            
            # Log successful loading
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logging.info(f"Model loaded successfully:")
            logging.info(f"  Total parameters: {total_params:,}")
            logging.info(f"  Trainable parameters: {trainable_params:,}")
            logging.info(f"  Device: {next(model.parameters()).device}")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {checkpoint_path}: {str(e)}")
    
    @staticmethod
    def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clean state dict from potential compilation artifacts."""
        cleaned_state_dict = {}
        
        for key, value in state_dict.items():
            # Remove compilation artifacts
            new_key = key.replace('_orig_mod.', '')
            
            # Handle other potential prefixes that might cause issues
            if new_key.startswith('model.'):
                # Keep model prefix as expected
                pass
            elif not new_key.startswith('model.') and ('encoder' in new_key or 'decoder' in new_key or 'head' in new_key):
                # Add model prefix if missing
                new_key = 'model.' + new_key
            
            cleaned_state_dict[new_key] = value
        
        return cleaned_state_dict
    
    @staticmethod
    def find_best_checkpoint(runs_dir: str = "runs", task_name: str = "Task001_FOMO1") -> str:
        """
        Find the best available checkpoint with improved scoring.
        """
        task_dir = Path(runs_dir) / task_name
        
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")
        
        checkpoint_candidates = []
        
        # Search for checkpoints
        for checkpoint_dir in task_dir.rglob("checkpoints"):
            best_ckpt = checkpoint_dir / "best.ckpt"
            
            if best_ckpt.exists():
                # Score based on directory structure and naming
                score = ModelLoader._score_checkpoint_path(str(checkpoint_dir.parent))
                checkpoint_candidates.append((score, str(best_ckpt), str(checkpoint_dir.parent)))
        
        if not checkpoint_candidates:
            raise FileNotFoundError(f"No checkpoints found in {task_dir}")
        
        # Sort by score (highest first)
        checkpoint_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Log found checkpoints
        logging.info(f"Found {len(checkpoint_candidates)} checkpoint candidates:")
        for score, ckpt_path, model_dir in checkpoint_candidates[:5]:  # Show top 5
            logging.info(f"  Score {score:3d}: {Path(model_dir).name}")
        
        return checkpoint_candidates[0][1]  # Return path of best checkpoint
    
    @staticmethod
    def _score_checkpoint_path(path: str) -> int:
        """Score checkpoint based on path components."""
        path_lower = path.lower()
        score = 0
        
        # Prefer non-ablation models
        if 'ablation' not in path_lower:
            score += 20
        
        # Prefer baseline and optimized models
        if 'baseline' in path_lower:
            score += 15
        if 'optimized' in path_lower:
            score += 15
        
        # Prefer fusion models over single encoder
        if 'fusion' in path_lower:
            score += 10
        
        # Architectural preferences
        if 'unet_xl' in path_lower:
            score += 8
        elif 'mednext' in path_lower:
            score += 6
        
        # Prefer certain fusion types
        if 'attention' in path_lower:
            score += 5
        elif 'masked_mean' in path_lower:
            score += 3
        
        # Penalize debug or test models
        if any(term in path_lower for term in ['debug', 'test', 'temp', 'xxx']):
            score -= 10
        
        return score


def get_available_checkpoints(runs_dir: str = "runs", task_name: str = "Task001_FOMO1") -> List[Dict[str, Any]]:
    """Get list of available checkpoints with metadata."""
    task_dir = Path(runs_dir) / task_name
    
    if not task_dir.exists():
        return []
    
    checkpoints = []
    
    for checkpoint_dir in task_dir.rglob("checkpoints"):
        best_ckpt = checkpoint_dir / "best.ckpt"
        
        if best_ckpt.exists():
            try:
                # Get basic info
                model_dir = checkpoint_dir.parent
                score = ModelLoader._score_checkpoint_path(str(model_dir))
                
                # Try to get model info (may fail for some checkpoints)
                try:
                    model_info = ModelLoader.detect_model_architecture(str(best_ckpt))
                    architecture = model_info.get('architecture', 'unknown')
                    encoder_type = model_info.get('encoder_type', 'unknown')
                    fusion_type = model_info.get('fusion_type', 'none')
                except:
                    architecture = 'unknown'
                    encoder_type = 'unknown'
                    fusion_type = 'unknown'
                
                checkpoints.append({
                    'path': str(best_ckpt),
                    'model_dir': str(model_dir),
                    'name': model_dir.name,
                    'score': score,
                    'architecture': architecture,
                    'encoder_type': encoder_type,
                    'fusion_type': fusion_type,
                    'file_size': best_ckpt.stat().st_size
                })
                
            except Exception as e:
                logging.debug(f"Failed to analyze checkpoint {best_ckpt}: {e}")
                continue
    
    # Sort by score
    checkpoints.sort(key=lambda x: x['score'], reverse=True)
    
    return checkpoints


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    try:
        checkpoints = get_available_checkpoints()

        logging.info(f"Found {len(checkpoints)} checkpoints")
        for ckpt in checkpoints[:10]:  # Show top 10
            logging.info(
                f"  {ckpt['score']:3d}: {ckpt['name']} ({ckpt['architecture']}, {ckpt['encoder_type']})"
            )

        if checkpoints:
            best_checkpoint = checkpoints[0]['path']
            logging.info(f"Best checkpoint: {best_checkpoint}")

            # Demonstrate model loading
            logging.info("Loading model...")
            _ = ModelLoader.load_model(best_checkpoint, 'classification')
            logging.info("Model loaded successfully!")
    
    except Exception as e:
        logging.error(f"Error: {e}")
