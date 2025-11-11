task1_config = {
    "task_name": "Task001_FOMO1",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    # Canonical 4-modality order for FOMO1 finetuning
    # [0] DWI, [1] ADC, [2] T2 FLAIR, [3] SWI or T2*
    "modalities": ("DWI", "ADC", "T2FLAIR", "SWI_OR_T2STAR"),
    "norm_op": "volume_wise_znorm",
    "num_classes": 2,
    "keep_aspect_ratio": True,
    "task_type": "classification",
    "label_extension": ".txt",
    "labels": {0: "Negative", 1: "Positive"},
    # Spacing and size configuration from training runs
    "target_spacing": [0.7188, 0.7188, 6.5000],  # Dominant modality spacing
    "target_size": None, # Critial: Must be set to None
    "patch_size": [96, 96, 96],
}

task2_config = {
    "task_name": "Task002_FOMO2",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    # Task 2 modalities: T2 FLAIR, DWI (b-value 1000), and either T2* or SWI images
    "modalities": ("DWI", "T2FLAIR", "SWI_OR_T2STAR"),
    "norm_op": "volume_wise_znorm", 
    "num_classes": 2,  # background (0) + meningioma (1)
    "num_modalities": 3,  # Number of input modalities
    "patch_size": [320, 320, 20],  # Corrected from training runs - was [64, 64, 30]
    "model_name": "unet_xl",  # Correct lowercase model name
    "version_dir": "./runs/Task002_FOMO2",  # Default output directory
    "keep_aspect_ratio": True,
    "task_type": "segmentation",  # Changed from classification to segmentation
    "label_extension": ".nii.gz",  # Changed from .txt to .nii.gz for segmentation masks
    "labels": {0: "background", 1: "meningioma"},  # Fixed typo: menigioma -> meningioma
    # Multi-encoder configuration for multi-modal fusion
    "use_multi_encoder": True,
    "multi_encoder_modalities": ["DWI", "T2FLAIR", "SWI_OR_T2STAR"],
    "modality_to_global_group": {"DWI": "dwi", "T2FLAIR": "flair", "SWI_OR_T2STAR": "other"},
    "global_vocab": ["t1", "t2", "flair", "dwi", "other"],
    "fusion_type": "attention",  # Use attention fusion for Task 2
    # Spacing and size configuration from training runs
    "target_spacing": [0.4500, 0.4500, 5.0000],  # From training runs
    "target_size": None, # Critial: Must be set to None
}

task3_config = {
    "task_name": "Task003_FOMO3",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("T1", "T2"),
    "norm_op": "volume_wise_znorm",
    "num_classes": 1,  # For regression, output dimension is 1
    "keep_aspect_ratio": True,
    "task_type": "regression",
    "label_extension": ".txt",
    "labels": {"regression": "Age"},  # Define as regression task
    # Spacing and size configuration
    "target_spacing": [0.5078, 0.7917, 6.0000],  # Already correctly configured
    "target_size": None, # Critial: Must be set to None
    "patch_size": [256, 256, 32],
}

hbn_config = {
    "task_name": "Task004_HBN",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("T1w",),  # Single T1w modality
    "norm_op": "volume_wise_znorm",
    "num_classes": 1,  # For regression, output dimension is 1
    "keep_aspect_ratio": True,
    "task_type": "regression",
    "label_extension": ".txt",
    "labels": {"regression": "Age"},  # Age regression task for HBN dataset
}
