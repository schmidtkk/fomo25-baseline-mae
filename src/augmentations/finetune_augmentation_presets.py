from typing import Literal


def get_finetune_augmentation_params(preset: Literal["basic", "anisotropic", "none", "yucca_default", "all"]) -> dict:
    """ "
    Get an augmentation parameter dict from a preset name.
    """
    if preset == "basic":
        return {
            # turned on
            "rotation_p_per_sample": 0.2,
            "rotation_p_per_axis": 0.66,
            "scale_p_per_sample": 0.2,
            "normalize": False,
            # turned off
            "additive_noise_p_per_sample": 0.0,
            "biasfield_p_per_sample": 0.0,
            "blurring_p_per_sample": 0.0,
            "blurring_p_per_channel": 0.0,
            "elastic_deform_p_per_sample": 0.0,
            "gamma_p_per_sample": 0.0,
            "gamma_p_invert_image": 0.0,
            "gibbs_ringing_p_per_sample": 0.0,
            "mirror_p_per_sample": 0.0,
            "mirror_p_per_axis": 0.0,
            "motion_ghosting_p_per_sample": 0.0,
            "multiplicative_noise_p_per_sample": 0.0,
            "simulate_lowres_p_per_sample": 0.0,
            "simulate_lowres_p_per_channel": 0.0,
            "simulate_lowres_p_per_axis": 0.0,
        }
    elif preset == "anisotropic":
        return {
            # Anisotropic-aware spatial augmentations
            "rotation_p_per_sample": 0.3,
            "rotation_p_per_axis": 0.5,  # Reduced from isotropic default
            "rotation_x": (-10, 10),     # Limited sagittal rotation for thick slices
            "rotation_y": (-10, 10),     # Limited coronal rotation for thick slices  
            "rotation_z": (-20, 20),     # More axial rotation allowed
            "scale_p_per_sample": 0.2,
            "scale_range_x": (0.9, 1.1), # Standard in-plane scaling
            "scale_range_y": (0.9, 1.1), # Standard in-plane scaling
            "scale_range_z": (0.98, 1.02), # Conservative through-plane scaling
            "elastic_deform_p_per_sample": 0.0,  # Disabled for thick-slice data
            "normalize": False,
            # turned off
            "additive_noise_p_per_sample": 0.0,
            "biasfield_p_per_sample": 0.0,
            "blurring_p_per_sample": 0.0,
            "blurring_p_per_channel": 0.0,
            "gamma_p_per_sample": 0.0,
            "gamma_p_invert_image": 0.0,
            "gibbs_ringing_p_per_sample": 0.0,
            "mirror_p_per_sample": 0.0,
            "mirror_p_per_axis": 0.0,
            "motion_ghosting_p_per_sample": 0.0,
            "multiplicative_noise_p_per_sample": 0.0,
            "simulate_lowres_p_per_sample": 0.0,
            "simulate_lowres_p_per_channel": 0.0,
            "simulate_lowres_p_per_axis": 0.0,
        }
    elif preset == "none":
        return {
            "rotation_p_per_sample": 0.0,
            "rotation_p_per_axis": 0.0,
            "scale_p_per_sample": 0.0,
            "additive_noise_p_per_sample": 0.0,
            "biasfield_p_per_sample": 0.0,
            "blurring_p_per_sample": 0.0,
            "blurring_p_per_channel": 0.0,
            "elastic_deform_p_per_sample": 0.0,
            "gamma_p_per_sample": 0.0,
            "gamma_p_invert_image": 0.0,
            "gibbs_ringing_p_per_sample": 0.0,
            "mirror_p_per_sample": 0.0,
            "mirror_p_per_axis": 0.0,
            "motion_ghosting_p_per_sample": 0.0,
            "multiplicative_noise_p_per_sample": 0.0,
            "simulate_lowres_p_per_sample": 0.0,
            "simulate_lowres_p_per_channel": 0.0,
            "simulate_lowres_p_per_axis": 0.0,
            "normalize": False,
        }
    elif preset == "all":
        return {"normalize": False}  # Will use Yucca defaults
    else:
        raise ValueError(f"Unknown augmentation preset: {preset}")
