import os
import SimpleITK as sitk
from dagster import asset

DATA_DIR = os.path.join("data", "images")
MASK_DIR = os.path.join("data", "masks")
ARTIFACTS_DIR = os.path.join("app", "storage", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

@asset
def raw_image() -> str:
    # Return existing image path
    path = os.path.normpath(os.path.join(DATA_DIR, "CT_pitch.nii.gz"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"raw_image: file not found at {path}")
    return path

@asset
def raw_mask() -> str:
    path = os.path.normpath(os.path.join(MASK_DIR, "CT_pitch_mask.nii.gz"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"raw_mask: file not found at {path}")
    return path

@asset
def registered_image(raw_image: str) -> str:
    # Read the raw image and write a registered copy
    img = sitk.ReadImage(raw_image)
    registered = sitk.Cast(img, img.GetPixelID())  # identity registration
    out_path = os.path.normpath(os.path.join(ARTIFACTS_DIR, "CT_pitch_registered.nii.gz"))
    sitk.WriteImage(registered, out_path)
    return out_path

@asset
def fused_image(registered_image: str, raw_mask: str) -> str:
    img = sitk.ReadImage(registered_image)
    mask = sitk.ReadImage(raw_mask)
    mask_uint8 = sitk.Cast(mask, sitk.sitkUInt8)
    fused = sitk.Mask(img, mask_uint8)
    out_path = os.path.normpath(os.path.join(ARTIFACTS_DIR, "CT_pitch_fused.nii.gz"))
    sitk.WriteImage(fused, out_path)
    return out_path

@asset
def features(fused_image: str) -> str:
    import json, numpy as np
    img = sitk.ReadImage(fused_image)
    arr = sitk.GetArrayFromImage(img)
    feats = {
        "shape": list(arr.shape),
        "spacing": list(img.GetSpacing()),
        "intensity_sum": float(np.sum(arr)),
    }
    out_path = os.path.normpath(os.path.join(ARTIFACTS_DIR, "CT_pitch_asset_features.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feats, f, indent=2)
    return out_path