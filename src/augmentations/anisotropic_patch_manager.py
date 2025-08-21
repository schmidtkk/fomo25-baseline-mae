"""
Anisotropic Patch Management for FOMO Tasks

This module handles patch extraction and processing for anisotropic medical imaging data.
It provides intelligent patch size selection and management that respects the different
spacing characteristics of the data.

Key Features:
- Task-specific patch size recommendations based on spacing analysis
- Anisotropic patch extraction that maintains anatomical context
- Memory-efficient patch processing for different anisotropy ratios
- Adaptive patch overlap strategies for thick-slice data

Designed for FOMO tasks:
- Task 1: 0.719×0.719×6.500mm → 96×96×24 patches
- Task 2: 0.859×0.859×6.500mm → 64×64×20 patches  
- Task 3: 0.508×0.792×6.000mm → 96×128×32 patches
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Dict, Any
import logging
from dataclasses import dataclass


@dataclass
class PatchConfig:
    """Configuration for anisotropic patch extraction."""
    patch_size: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    overlap_factor: Tuple[float, float, float] = (0.5, 0.5, 0.25)  # Less overlap in Z
    min_patch_size: Tuple[int, int, int] = (32, 32, 8)
    max_patch_size: Tuple[int, int, int] = (192, 192, 64)
    
    @property
    def anisotropy_ratio(self) -> float:
        """Calculate anisotropy ratio (max_spacing / min_spacing)."""
        return max(self.spacing) / min(self.spacing)
    
    @property
    def physical_patch_size(self) -> Tuple[float, float, float]:
        """Physical patch size in mm."""
        return tuple(p * s for p, s in zip(self.patch_size, self.spacing))


class AnisotropicPatchManager:
    """
    Manages patch extraction and processing for anisotropic medical imaging data.
    
    This class provides intelligent patch management that adapts to the spacing
    characteristics of different medical imaging protocols.
    """
    
    def __init__(self, task_id: int, custom_config: Optional[PatchConfig] = None):
        """
        Initialize patch manager for specific FOMO task.
        
        Args:
            task_id: FOMO task ID (1, 2, or 3)
            custom_config: Optional custom patch configuration
        """
        self.task_id = task_id
        self.config = custom_config or self._get_default_config(task_id)
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self, task_id: int) -> PatchConfig:
        """Get default patch configuration for task."""
        configs = {
            1: PatchConfig(  # Task 1: Stroke - highly anisotropic
                patch_size=(96, 96, 24),
                spacing=(0.719, 0.719, 6.500),
                overlap_factor=(0.5, 0.5, 0.2),  # Minimal Z overlap
            ),
            2: PatchConfig(  # Task 2: Meningioma - highly anisotropic
                patch_size=(64, 64, 20), 
                spacing=(0.859, 0.859, 6.500),
                overlap_factor=(0.5, 0.5, 0.25),
            ),
            3: PatchConfig(  # Task 3: Brain Age - variable protocols
                patch_size=(96, 128, 32),
                spacing=(0.508, 0.792, 6.000),
                overlap_factor=(0.4, 0.5, 0.3),  # More flexible
            ),
        }
        
        if task_id not in configs:
            self.logger.warning(f"Unknown task_id {task_id}, using default config")
            return PatchConfig(
                patch_size=(64, 64, 64),  # Isotropic fallback
                spacing=(1.0, 1.0, 1.0),
            )
            
        return configs[task_id]
    
    def suggest_patch_size(
        self, 
        data_shape: Tuple[int, int, int], 
        spacing: Optional[Tuple[float, float, float]] = None,
        target_physical_size: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[int, int, int]:
        """
        Suggest optimal patch size based on data characteristics.
        
        Args:
            data_shape: Shape of the input data (D, H, W)
            spacing: Voxel spacing (optional, uses task default if None)
            target_physical_size: Target physical patch size in mm (optional)
            
        Returns:
            Suggested patch size as (x, y, z) tuple
        """
        if spacing is None:
            spacing = self.config.spacing
            
        # Update config spacing if provided
        config = PatchConfig(
            patch_size=self.config.patch_size,
            spacing=spacing,
            overlap_factor=self.config.overlap_factor
        )
        
        if target_physical_size is not None:
            # Calculate patch size to achieve target physical size
            suggested = tuple(
                int(np.ceil(target / space)) 
                for target, space in zip(target_physical_size, spacing)
            )
        else:
            # Use heuristics based on anisotropy ratio
            aniso_ratio = config.anisotropy_ratio
            
            if aniso_ratio > 5:  # Highly anisotropic (Tasks 1&2)
                # Prioritize in-plane resolution, limit Z dimension
                z_patches = min(data_shape[2], max(8, data_shape[2] // 4))
                xy_patches = min(96, max(data_shape[0] // 4, data_shape[1] // 4))
                suggested = (xy_patches, xy_patches, z_patches)
                
            elif aniso_ratio > 2:  # Moderately anisotropic (Task 3)
                # Balance between dimensions
                x_patches = min(96, max(data_shape[0] // 3, 32))
                y_patches = min(128, max(data_shape[1] // 3, 32))  
                z_patches = min(64, max(data_shape[2] // 2, 16))
                suggested = (x_patches, y_patches, z_patches)
                
            else:  # Nearly isotropic
                # Use balanced patch size
                patch_size = min(64, max(32, min(data_shape) // 3))
                suggested = (patch_size, patch_size, patch_size)
        
        # Ensure patch size is within reasonable bounds
        suggested = tuple(
            max(min_s, min(max_s, s)) 
            for s, min_s, max_s in zip(suggested, config.min_patch_size, config.max_patch_size)
        )
        
        # Ensure patch size is not larger than data
        suggested = tuple(
            min(s, d) for s, d in zip(suggested, data_shape)
        )
        
        return suggested
    
    def calculate_patch_positions(
        self, 
        data_shape: Tuple[int, int, int],
        patch_size: Optional[Tuple[int, int, int]] = None,
        overlap_factor: Optional[Tuple[float, float, float]] = None
    ) -> List[Tuple[slice, slice, slice]]:
        """
        Calculate patch extraction positions with anisotropic overlap.
        
        Args:
            data_shape: Shape of the input data
            patch_size: Patch size (uses config default if None)
            overlap_factor: Overlap factor per axis (uses config default if None)
            
        Returns:
            List of slice tuples for patch extraction
        """
        if patch_size is None:
            patch_size = self.config.patch_size
        if overlap_factor is None:
            overlap_factor = self.config.overlap_factor
            
        positions = []
        
        for dim in range(3):
            data_size = data_shape[dim]
            patch_dim = patch_size[dim]
            overlap = overlap_factor[dim]
            
            if data_size <= patch_dim:
                # Data smaller than patch, use single centered patch
                start = max(0, (data_size - patch_dim) // 2)
                positions.append([slice(start, start + patch_dim)])
            else:
                # Calculate step size with overlap
                step = int(patch_dim * (1 - overlap))
                step = max(1, step)  # Ensure minimum step
                
                dim_positions = []
                start = 0
                while start < data_size:
                    end = min(start + patch_dim, data_size)
                    if end - start < patch_dim // 2:
                        # If remaining patch is too small, extend previous patch
                        if dim_positions:
                            dim_positions[-1] = slice(data_size - patch_dim, data_size)
                        break
                    else:
                        dim_positions.append(slice(start, end))
                        start += step
                        
                positions.append(dim_positions)
        
        # Generate all combinations of positions
        patch_positions = []
        for x_slice in positions[0]:
            for y_slice in positions[1]:
                for z_slice in positions[2]:
                    patch_positions.append((x_slice, y_slice, z_slice))
                    
        return patch_positions
    
    def extract_patch(
        self, 
        data: np.ndarray, 
        position: Tuple[slice, slice, slice],
        pad_mode: str = 'constant'
    ) -> np.ndarray:
        """
        Extract a patch from data at the specified position.
        
        Args:
            data: Input data array
            position: Patch position as (x_slice, y_slice, z_slice)
            pad_mode: Padding mode if patch extends beyond data
            
        Returns:
            Extracted patch
        """
        if len(data.shape) == 4:  # Multi-channel data (C, D, H, W)
            return data[:, position[0], position[1], position[2]]
        else:  # Single-channel data (D, H, W)
            return data[position[0], position[1], position[2]]
    
    def reconstruct_from_patches(
        self,
        patches: List[np.ndarray],
        positions: List[Tuple[slice, slice, slice]],
        data_shape: Tuple[int, int, int],
        overlap_mode: str = 'average'
    ) -> np.ndarray:
        """
        Reconstruct full volume from overlapping patches.
        
        Args:
            patches: List of patch arrays
            positions: List of patch positions
            data_shape: Shape of the target volume
            overlap_mode: How to handle overlapping regions ('average', 'maximum')
            
        Returns:
            Reconstructed volume
        """
        # Determine output shape (handle multi-channel data)
        if len(patches[0].shape) == 4:  # Multi-channel patches
            output_shape = (patches[0].shape[0], *data_shape)
            reconstruction = np.zeros(output_shape)
            count_map = np.zeros(output_shape)
        else:  # Single-channel patches
            reconstruction = np.zeros(data_shape)
            count_map = np.zeros(data_shape)
        
        # Accumulate patches
        for patch, position in zip(patches, positions):
            if len(patch.shape) == 4:  # Multi-channel
                reconstruction[:, position[0], position[1], position[2]] += patch
                count_map[:, position[0], position[1], position[2]] += 1
            else:  # Single-channel
                reconstruction[position[0], position[1], position[2]] += patch
                count_map[position[0], position[1], position[2]] += 1
        
        # Handle overlapping regions
        if overlap_mode == 'average':
            # Avoid division by zero
            count_map[count_map == 0] = 1
            reconstruction = reconstruction / count_map
        elif overlap_mode == 'maximum':
            # Keep maximum values (useful for segmentation)
            pass  # Already handled by accumulation
            
        return reconstruction
    
    def get_memory_efficient_batch_size(
        self,
        available_memory_gb: float = 8.0,
        data_dtype: np.dtype = np.float32
    ) -> int:
        """
        Calculate memory-efficient batch size based on patch configuration.
        
        Args:
            available_memory_gb: Available GPU memory in GB
            data_dtype: Data type of input arrays
            
        Returns:
            Suggested batch size
        """
        # Calculate memory per patch (including gradients and intermediate activations)
        patch_volume = np.prod(self.config.patch_size)
        bytes_per_element = np.dtype(data_dtype).itemsize
        
        # Rough estimation: input + gradients + activations ≈ 4x input size
        memory_per_patch_mb = (patch_volume * bytes_per_element * 4) / (1024 ** 2)
        available_memory_mb = available_memory_gb * 1024
        
        # Reserve 20% for other operations
        usable_memory_mb = available_memory_mb * 0.8
        
        batch_size = max(1, int(usable_memory_mb / memory_per_patch_mb))
        
        # Cap batch size for very small patches
        max_batch_size = 16 if self.config.anisotropy_ratio > 5 else 8
        batch_size = min(batch_size, max_batch_size)
        
        return batch_size
    
    def log_configuration(self):
        """Log current patch configuration."""
        self.logger.info(f"Patch Manager Configuration for Task {self.task_id}:")
        self.logger.info(f"  Patch size: {self.config.patch_size}")
        self.logger.info(f"  Spacing: {self.config.spacing}")
        self.logger.info(f"  Physical patch size: {self.config.physical_patch_size}")
        self.logger.info(f"  Anisotropy ratio: {self.config.anisotropy_ratio:.2f}")
        self.logger.info(f"  Overlap factor: {self.config.overlap_factor}")


def get_task_patch_config(task_id: int) -> Dict[str, Any]:
    """
    Get patch configuration for finetune scripts.
    
    Args:
        task_id: FOMO task ID
        
    Returns:
        Dictionary with patch configuration parameters
    """
    manager = AnisotropicPatchManager(task_id)
    
    return {
        'patch_size': manager.config.patch_size,
        'spacing': manager.config.spacing,
        'overlap_factor': manager.config.overlap_factor,
        'anisotropy_ratio': manager.config.anisotropy_ratio,
        'physical_size': manager.config.physical_patch_size,
        'suggested_batch_size': manager.get_memory_efficient_batch_size(),
    }


# Example usage and testing
if __name__ == "__main__":
    # Test for each FOMO task
    for task_id in [1, 2, 3]:
        print(f"\n=== Task {task_id} Configuration ===")
        
        manager = AnisotropicPatchManager(task_id)
        manager.log_configuration()
        
        # Test patch position calculation
        data_shape = (256, 256, 24)  # Typical anisotropic volume
        positions = manager.calculate_patch_positions(data_shape)
        print(f"Number of patches: {len(positions)}")
        print(f"First few positions: {positions[:3]}")
        
        # Test patch size suggestion
        suggested = manager.suggest_patch_size(data_shape)
        print(f"Suggested patch size: {suggested}")
        
        # Test memory calculation
        batch_size = manager.get_memory_efficient_batch_size()
        print(f"Suggested batch size: {batch_size}")
