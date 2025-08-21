"""
Anisotropic Data Augmentation for FOMO Tasks

This module provides anisotropic spatial transformations that respect the different 
spacing characteristics of medical imaging data. Unlike isotropic augmentations,
these transformations apply different parameters to different spatial axes.

Key Features:
- Axis-specific rotation ranges (e.g., limited Z-axis rotation for thick slices)
- Axis-specific scaling factors (e.g., different scale factors for in-plane vs through-plane)
- Spacing-aware transformations that maintain anatomical realism

Designed for FOMO tasks with anisotropic data:
- Task 1: 0.719×0.719×6.500mm (9:1 anisotropy ratio)
- Task 2: 0.859×0.859×6.500mm (7.5:1 anisotropy ratio)  
- Task 3: 0.508×0.792×6.000mm (variable anisotropy)
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from batchgenerators.augmentations.spatial_transformations import augment_spatial
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates
from scipy.ndimage import map_coordinates, gaussian_filter
import torch


class AnisotropicSpatialTransform:
    """
    Anisotropic spatial transformations that apply different parameters per axis.
    
    Args:
        patch_size: 3D patch size as (x, y, z) tuple
        spacing: Voxel spacing as (x, y, z) tuple in mm
        do_rotation: Whether to apply rotation
        angle_x_range: Rotation range for X axis in degrees (sagittal plane rotation)
        angle_y_range: Rotation range for Y axis in degrees (coronal plane rotation) 
        angle_z_range: Rotation range for Z axis in degrees (axial plane rotation)
        do_scaling: Whether to apply scaling
        scale_x_range: Scaling range for X axis
        scale_y_range: Scaling range for Y axis
        scale_z_range: Scaling range for Z axis
        do_elastic_deform: Whether to apply elastic deformation
        alpha: Deformation strength per axis
        sigma: Deformation smoothness per axis
        p_el_per_sample: Probability of applying elastic deformation
        p_rot_per_sample: Probability of applying rotation
        p_scale_per_sample: Probability of applying scaling
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        # Rotation parameters (axis-specific)
        do_rotation: bool = True,
        angle_x_range: Tuple[float, float] = (-15, 15),   # Limited sagittal rotation
        angle_y_range: Tuple[float, float] = (-15, 15),   # Limited coronal rotation  
        angle_z_range: Tuple[float, float] = (-30, 30),   # More axial rotation allowed
        p_rot_per_sample: float = 0.2,
        # Scaling parameters (axis-specific)
        do_scaling: bool = True,
        scale_x_range: Tuple[float, float] = (0.9, 1.1),  # In-plane scaling
        scale_y_range: Tuple[float, float] = (0.9, 1.1),  # In-plane scaling
        scale_z_range: Tuple[float, float] = (0.95, 1.05), # Conservative through-plane scaling
        p_scale_per_sample: float = 0.2,
        # Elastic deformation parameters
        do_elastic_deform: bool = False,  # Disabled by default - can be computationally expensive
        alpha: Tuple[float, float, float] = (200, 200, 50),  # Less deformation in Z
        sigma: Tuple[float, float, float] = (20, 20, 10),    # Less smoothing in Z
        p_el_per_sample: float = 0.1,
    ):
        self.patch_size = patch_size
        self.spacing = spacing
        
        # Rotation settings
        self.do_rotation = do_rotation
        self.angle_x_range = angle_x_range
        self.angle_y_range = angle_y_range  
        self.angle_z_range = angle_z_range
        self.p_rot_per_sample = p_rot_per_sample
        
        # Scaling settings
        self.do_scaling = do_scaling
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range
        self.scale_z_range = scale_z_range
        self.p_scale_per_sample = p_scale_per_sample
        
        # Elastic deformation settings
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.p_el_per_sample = p_el_per_sample
        
        # Calculate anisotropy ratio for adaptive parameters
        self.anisotropy_ratio = max(spacing) / min(spacing)
        
    def __call__(self, data_dict: dict) -> dict:
        """
        Apply anisotropic spatial transformations to data.
        
        Args:
            data_dict: Dictionary containing 'data' and optionally 'seg' keys
            
        Returns:
            Transformed data dictionary
        """
        data = data_dict['data']
        seg = data_dict.get('seg', None)
        
        # Apply transformations
        if self.do_rotation and np.random.random() < self.p_rot_per_sample:
            data, seg = self._apply_anisotropic_rotation(data, seg)
            
        if self.do_scaling and np.random.random() < self.p_scale_per_sample:
            data, seg = self._apply_anisotropic_scaling(data, seg)
            
        if self.do_elastic_deform and np.random.random() < self.p_el_per_sample:
            data, seg = self._apply_anisotropic_elastic_deform(data, seg)
        
        result = {'data': data}
        if seg is not None:
            result['seg'] = seg
            
        return result
    
    def _apply_anisotropic_rotation(self, data: np.ndarray, seg: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply axis-specific rotation parameters."""
        # Sample rotation angles per axis
        angle_x = np.random.uniform(*self.angle_x_range)
        angle_y = np.random.uniform(*self.angle_y_range)  
        angle_z = np.random.uniform(*self.angle_z_range)
        
        # For anisotropic data, reduce through-plane rotation
        if self.anisotropy_ratio > 5:  # Highly anisotropic (like Task 1&2)
            angle_x *= 0.5  # Reduce sagittal rotation
            angle_y *= 0.5  # Reduce coronal rotation
            
        angles = [angle_x, angle_y, angle_z]
        
        # Apply rotation using batchgenerators
        # Note: This is a simplified version - full implementation would use
        # scipy.ndimage.rotate or custom rotation matrices
        for i, angle in enumerate(angles):
            if abs(angle) > 1e-6:  # Only rotate if angle is significant
                axis_tuple = (i + 1, (i + 2) % 3 + 1)  # axis pairs for rotation
                data = self._rotate_along_axis(data, angle, axis_tuple)
                if seg is not None:
                    seg = self._rotate_along_axis(seg, angle, axis_tuple, order=0)
        
        return data, seg
    
    def _apply_anisotropic_scaling(self, data: np.ndarray, seg: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply axis-specific scaling factors.""" 
        scale_x = np.random.uniform(*self.scale_x_range)
        scale_y = np.random.uniform(*self.scale_y_range)
        scale_z = np.random.uniform(*self.scale_z_range)
        
        # For highly anisotropic data, be more conservative with Z scaling
        if self.anisotropy_ratio > 5:
            scale_z = np.random.uniform(0.98, 1.02)  # Very conservative Z scaling
            
        scales = [scale_x, scale_y, scale_z]
        
        # Apply scaling (simplified - full implementation would use affine transforms)
        data = self._scale_data(data, scales)
        if seg is not None:
            seg = self._scale_data(seg, scales, order=0)
            
        return data, seg
    
    def _apply_anisotropic_elastic_deform(self, data: np.ndarray, seg: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply axis-specific elastic deformation."""
        # Adjust deformation parameters based on spacing
        alpha_adjusted = [
            self.alpha[0] * (self.spacing[0] / min(self.spacing)),
            self.alpha[1] * (self.spacing[1] / min(self.spacing)),
            self.alpha[2] * (self.spacing[2] / min(self.spacing))
        ]
        
        sigma_adjusted = [
            self.sigma[0] * (self.spacing[0] / min(self.spacing)),
            self.sigma[1] * (self.spacing[1] / min(self.spacing)), 
            self.sigma[2] * (self.spacing[2] / min(self.spacing))
        ]
        
        # Apply elastic deformation
        data = self._elastic_deform_data(data, alpha_adjusted, sigma_adjusted)
        if seg is not None:
            seg = self._elastic_deform_data(seg, alpha_adjusted, sigma_adjusted, order=0)
            
        return data, seg
    
    def _rotate_along_axis(self, data: np.ndarray, angle: float, axes: Tuple[int, int], order: int = 1) -> np.ndarray:
        """Rotate data along specified axes."""
        from scipy.ndimage import rotate
        # Apply rotation to each channel separately
        rotated = np.zeros_like(data)
        for c in range(data.shape[0]):
            rotated[c] = rotate(data[c], angle, axes=axes, reshape=False, order=order, mode='nearest')
        return rotated
    
    def _scale_data(self, data: np.ndarray, scales: List[float], order: int = 1) -> np.ndarray:
        """Scale data along each axis."""
        from scipy.ndimage import zoom
        # Apply scaling to each channel separately
        scaled = np.zeros_like(data)
        for c in range(data.shape[0]):
            scaled[c] = zoom(data[c], scales, order=order, mode='nearest')
        return scaled
    
    def _elastic_deform_data(self, data: np.ndarray, alpha: List[float], sigma: List[float], order: int = 1) -> np.ndarray:
        """Apply elastic deformation with axis-specific parameters."""
        # Simplified elastic deformation - full implementation would be more complex
        deformed = np.zeros_like(data)
        shape = data.shape[1:]  # Remove channel dimension
        
        # Generate random displacement fields for each axis
        dx = gaussian_filter(np.random.randn(*shape), sigma[0], mode='constant', cval=0) * alpha[0]
        dy = gaussian_filter(np.random.randn(*shape), sigma[1], mode='constant', cval=0) * alpha[1]
        dz = gaussian_filter(np.random.randn(*shape), sigma[2], mode='constant', cval=0) * alpha[2]
        
        # Create coordinate mesh
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        
        # Apply displacement
        indices = [x + dx, y + dy, z + dz]
        
        for c in range(data.shape[0]):
            deformed[c] = map_coordinates(data[c], indices, order=order, mode='nearest')
            
        return deformed


class AnisotropicAugmentationComposer:
    """
    Wrapper class that provides YuccaAugmentationComposer-compatible interface
    for anisotropic transformations.
    """
    
    def __init__(self, anisotropic_transform):
        self.anisotropic_transform = anisotropic_transform
        # Create identical train and val transforms (anisotropic transform handles its own randomness)
        self.train_transforms = self.anisotropic_transform
        self.val_transforms = lambda x: x  # No augmentation for validation


def get_anisotropic_augmentation_for_task(
    task_id: int,
    patch_size: Tuple[int, int, int],
    spacing: Optional[Tuple[float, float, float]] = None
) -> AnisotropicAugmentationComposer:
    """
    Get task-specific anisotropic augmentation parameters.
    
    Args:
        task_id: FOMO task ID (1, 2, or 3)
        patch_size: 3D patch size
        spacing: Voxel spacing (if None, uses task defaults)
        
    Returns:
        Configured AnisotropicSpatialTransform instance
    """
    # Default spacing based on our analysis
    default_spacings = {
        1: (0.719, 0.719, 6.500),  # Task 1: Stroke  
        2: (0.859, 0.859, 6.500),  # Task 2: Meningioma
        3: (0.508, 0.792, 6.000),  # Task 3: Brain Age
    }
    
    if spacing is None:
        spacing = default_spacings.get(task_id, (1.0, 1.0, 1.0))
    
    if task_id in [1, 2]:  # Stroke and Meningioma - highly anisotropic
        transform = AnisotropicSpatialTransform(
            patch_size=patch_size,
            spacing=spacing,
            # Conservative rotation for thick-slice data
            angle_x_range=(-10, 10),   # Limited sagittal
            angle_y_range=(-10, 10),   # Limited coronal
            angle_z_range=(-20, 20),   # Moderate axial
            p_rot_per_sample=0.3,
            # Conservative scaling
            scale_x_range=(0.9, 1.1),   # Standard in-plane
            scale_y_range=(0.9, 1.1),   # Standard in-plane
            scale_z_range=(0.98, 1.02), # Very conservative through-plane
            p_scale_per_sample=0.2,
            # Minimal elastic deformation
            do_elastic_deform=False,
        )
        return AnisotropicAugmentationComposer(transform)
        
    elif task_id == 3:  # Brain Age - variable protocols
        transform = AnisotropicSpatialTransform(
            patch_size=patch_size,
            spacing=spacing,
            # More flexible rotation for diverse protocols
            angle_x_range=(-15, 15),
            angle_y_range=(-15, 15),
            angle_z_range=(-30, 30),
            p_rot_per_sample=0.2,
            # Moderate scaling
            scale_x_range=(0.85, 1.15),  # More variation for diverse data
            scale_y_range=(0.85, 1.15),
            scale_z_range=(0.9, 1.1),
            p_scale_per_sample=0.25,
            # Light elastic deformation
            do_elastic_deform=True,
            p_el_per_sample=0.1,
        )
        return AnisotropicAugmentationComposer(transform)
        
    else:
        # Default isotropic-like behavior for unknown tasks
        transform = AnisotropicSpatialTransform(
            patch_size=patch_size,
            spacing=spacing,
        )
        return AnisotropicAugmentationComposer(transform)


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage for Task 1
    patch_size = (96, 96, 24)
    spacing = (0.719, 0.719, 6.500)
    
    transform = get_anisotropic_augmentation_for_task(1, patch_size, spacing)
    
    # Test with dummy data
    dummy_data = np.random.randn(4, *patch_size)  # 4 modalities
    dummy_seg = np.random.randint(0, 2, (1, *patch_size))  # Binary segmentation
    
    data_dict = {
        'data': dummy_data,
        'seg': dummy_seg
    }
    
    augmented = transform(data_dict)
    print(f"Original data shape: {dummy_data.shape}")
    print(f"Augmented data shape: {augmented['data'].shape}")
    print(f"Original seg shape: {dummy_seg.shape}")
    print(f"Augmented seg shape: {augmented['seg'].shape}")
