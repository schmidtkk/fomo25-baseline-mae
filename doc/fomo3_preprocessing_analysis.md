# FOMO3 Preprocessing Pipeline Analysis

## Overview
This document provides a detailed analysis of the preprocessing pipeline for Task 3 (FOMO3) in the baseline codebase. The preprocessing pipeline is responsible for preparing multi-modal medical imaging data (T1 and T2 modalities) for training and inference.

---

## 1. How does the npy output get ranged to 0~1?

The normalization to the 0-1 range is achieved through the **`volume_wise_znorm`** operation in the yucca preprocessing pipeline. The steps are as follows:

1. **Clipping Outliers**:
   - Values are clipped to the 99th percentile to remove extreme outliers.
   ```python
   q_val = np.quantile(x[mask], 0.99)
   return np.clip(x, a_min=None, a_max=q_val)
   ```

2. **Z-Normalization**:
   - The mean and standard deviation of the non-background values are computed, and the data is normalized to have zero mean and unit variance.
   ```python
   values = x[mask]
   mean, std = np.mean(values), np.std(values)
   x -= mean
   x /= std
   ```

3. **Rescaling to [0,1]**:
   - The normalized data is rescaled to the range [0,1] using scikit-image's `rescale_intensity` function.
   ```python
   return exposure.rescale_intensity(x, out_range=(0, 1))
   ```

**Evidence from Data:**
- Original T1: min=0.0, max=208.6, mean=15.7, std=32.4
- Processed T1: min=0.0, max=1.0, mean=0.29, std=0.36
- Original T2: min=0.0, max=255.0, mean=9.1, std=20.1
- Processed T2: min=0.0, max=1.0, mean=0.16, std=0.22

---

## 2. What does yucca preproc do to the .nii.gz file of Task 3 dataset?

The FOMO3 preprocessing pipeline (`fomo3_fusion.py`) performs the following operations:

### A. **Multi-modal Detection & Loading**
- Detects T1 and T2 modalities by filename pattern matching.
- Loads the NIfTI files using nibabel.

### B. **Spatial Preprocessing via Yucca**
- The `preprocess_case_for_training_without_label` function is used to preprocess the data:
  ```python
  preprocessed_images, _ = preprocess_case_for_training_without_label(
      images=[t1_nii, t2_nii],
      normalization_operation=["volume_wise_znorm", "volume_wise_znorm"],
      crop_to_nonzero=True,
      target_spacing=[0.5078, 0.7917, 6.0000],
      target_size=[256, 256, 32]
  )
  ```

**Steps:**
1. **Resampling**: Changes voxel spacing from (0.859, 0.859, 6.0) to target (0.508, 0.792, 6.0).
2. **Resizing**: Changes volume dimensions from (256, 256, 26) to (256, 256, 32).
3. **Cropping**: Removes zero-valued background regions.
4. **Alignment**: Ensures spatial correspondence between T1 and T2 modalities.
5. **Normalization**: Applies `volume_wise_znorm` to each modality.

### C. **Output Structure Creation**
For each subject, the following files are created:
```
FOMO3_sub_1/
├── T1.npy        # Preprocessed T1 volume [256,256,32] float32 [0,1]
├── T2.npy        # Preprocessed T2 volume [256,256,32] float32 [0,1]  
├── mask.json     # {"T1": 1, "T2": 1} - modality presence
└── label.txt     # Age value (e.g., "30")
```

### D. **Key Transformations Summary**
- **Spatial**: (256,256,26) @ (0.859,0.859,6.0)mm → (256,256,32) @ (0.508,0.792,6.0)mm
- **Intensity**: Raw intensity values → Z-normalized → [0,1] range
- **Format**: .nii.gz → .npy (faster loading during training)
- **Multi-modal**: Joint spatial alignment of T1/T2 modalities
- **Fusion-ready**: Per-modality files + presence mask for flexible fusion strategies

---

## Conclusion
The preprocessing pipeline ensures that the data is spatially aligned, intensity-normalized, and formatted for efficient training. The output structure and transformations are optimized for multi-modal fusion strategies, making the data ready for downstream tasks like regression or segmentation.
