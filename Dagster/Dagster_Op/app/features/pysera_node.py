import os
import json
import numpy as np
import SimpleITK as sitk

def run_pysera(image_path: str, mask_path: str):
    # Read image and mask from file paths
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))

    # Ensure mask is binary {0,1} and integer typed
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    # Treat any non-zero voxel as 1
    mask = sitk.BinaryThreshold(
        mask,
        lowerThreshold=1,
        upperThreshold=1_000_000,
        insideValue=1,
        outsideValue=0,
    )

    # Use proper radius vector for BinaryErode based on image dimension
    dim = image.GetDimension()
    radius = [1] * dim  # e.g., [1,1,1] for 3D or [1,1] for 2D

    # Safe morphological operation on the binary mask
    eroded = sitk.BinaryErode(mask, radius)
    mask_array = sitk.GetArrayFromImage(mask)
    eroded_array = sitk.GetArrayFromImage(eroded)
    mask_border = np.logical_xor(mask_array, eroded_array)

    # Example features (replace with your real extraction pipeline)
    features = {
        "voxel_spacing": tuple(image.GetSpacing()),
        "mask_voxels": int(np.sum(mask_array)),
        "mask_border_voxels": int(np.sum(mask_border)),
    }

    return features
