import os
import pathlib
import time
import json
import sys
from typing import List, Dict

import pandas as pd
import SimpleITK as sitk
from dagster import asset, IOManager, io_manager
import pysera

# Ensure stdout can handle UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join("data", "images")
MASK_DIR = os.path.join("data", "masks")
ARTIFACTS_DIR = os.path.join("artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# JSON-safe conversion helper
def convert_paths_and_dfs(obj):
    # Convert nested objects to JSON-safe types
    if isinstance(obj, dict):
        return {k: convert_paths_and_dfs(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_and_dfs(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_paths_and_dfs(v) for v in obj]
    elif isinstance(obj, pathlib.Path):
        return str(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    else:
        return obj

# Custom IOManager: persist asset outputs as JSON in artifacts/
class JsonFileIOManager(IOManager):
    def handle_output(self, context, obj):
        file_path = os.path.join(ARTIFACTS_DIR, f"{context.asset_key.path[-1]}.json")
        safe_obj = convert_paths_and_dfs(obj)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(safe_obj, f, indent=2, ensure_ascii=False)
        context.log.info(f"Output written to {file_path}")

    def load_input(self, context):
        file_path = os.path.join(ARTIFACTS_DIR, f"{context.upstream_output.asset_key.path[-1]}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        context.log.info(f"Input loaded from {file_path}")
        return data

@io_manager
def json_io_manager(_):
    return JsonFileIOManager()

# Masks discovery
@asset
def all_masks() -> List[str]:
    files = [os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith(".nii.gz")]
    if not files:
        raise FileNotFoundError("No masks found in data/masks")
    return files

# Image reader (force float32 and persist)
@asset
def image_reader() -> List[str]:
    reader_dir = os.path.join(ARTIFACTS_DIR, "reader")
    os.makedirs(reader_dir, exist_ok=True)

    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".nii.gz")]
    if not files:
        raise FileNotFoundError("No images found in data/images")

    converted_paths = []
    for path in files:
        img = sitk.ReadImage(path)
        print(f"[reader] raw {os.path.basename(path)} -> dim={img.GetDimension()}, type={img.GetPixelIDTypeAsString()}")
        img_float = sitk.Cast(img, sitk.sitkFloat32)
        print(f"[reader] casted {os.path.basename(path)} -> type={img_float.GetPixelIDTypeAsString()}")
        out_path = os.path.join(reader_dir, f"reader_{os.path.basename(path)}")
        sitk.WriteImage(img_float, out_path)
        converted_paths.append(out_path)

    return converted_paths


# Utilities for registration and I/O
def write_nifti(image: sitk.Image, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(image, out_path)

def cast_to_float32(img: sitk.Image, label: str) -> sitk.Image:
    casted = sitk.Cast(img, sitk.sitkFloat32)
    print(f"[registration] {label}: dim={casted.GetDimension()}, type={casted.GetPixelIDTypeAsString()}")
    return casted

def make_initial_transform(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    dim = fixed.GetDimension()
    if dim == 2:
        return sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif dim == 3:
        return sitk.CenteredTransformInitializer(
            fixed, moving, sitk.VersorRigid3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    else:
        raise RuntimeError(f"Unsupported image dimension: {dim}")

# Image registration (robust casting + 2D/3D support)
@asset
def image_registration(image_reader: List[str]) -> List[str]:
    fixed_raw = sitk.ReadImage(image_reader[0])
    print(f"[registration] fixed_raw: dim={fixed_raw.GetDimension()}, type={fixed_raw.GetPixelIDTypeAsString()}")
    fixed = cast_to_float32(fixed_raw, "fixed_cast")

    R = sitk.ImageRegistrationMethod()
    if fixed.GetDimension() == 2:
        R.SetMetricAsMeanSquares()
        R.SetInterpolator(sitk.sitkLinear)
    else:
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.2)
        R.SetInterpolator(sitk.sitkLinear)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200, gradientMagnitudeTolerance=1e-8
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    out_paths = []
    for img_path in image_reader:
        moving_raw = sitk.ReadImage(img_path)
        print(f"[registration] moving_raw: {os.path.basename(img_path)} dim={moving_raw.GetDimension()}, type={moving_raw.GetPixelIDTypeAsString()}")
        moving = cast_to_float32(moving_raw, f"moving_cast:{os.path.basename(img_path)}")

        init_tx = make_initial_transform(fixed, moving)
        R.SetInitialTransform(init_tx, inPlace=False)

        final_tx = R.Execute(fixed, moving)
        registered = sitk.Resample(moving, fixed, final_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

        out_path = os.path.join(ARTIFACTS_DIR, f"registered_{os.path.basename(img_path)}")
        write_nifti(registered, out_path)
        out_paths.append(out_path)

    return out_paths


# Fusion (robust intensity normalization)
@asset
def image_fusion(image_registration: List[str]) -> List[str]:
    fused_paths = []
    import numpy as np
    for img_path in image_registration:
        img = sitk.ReadImage(img_path)
        arr = sitk.GetArrayFromImage(img)
        p5, p95 = np.percentile(arr, [5, 95])
        arr = np.clip(arr, p5, p95)
        arr = (arr - p5) / (p95 - p5) if p95 > p5 else arr * 0.0
        fused_img = sitk.GetImageFromArray(arr)
        fused_img.CopyInformation(img)
        out_path = os.path.join(ARTIFACTS_DIR, f"fused_{os.path.basename(img_path)}")
        write_nifti(fused_img, out_path)
        fused_paths.append(out_path)
    return fused_paths


# Conversion (ensures consistent naming)
@asset
def image_conversion(image_fusion: List[str]) -> List[str]:
    converted_paths = []
    for img_path in image_fusion:
        img = sitk.ReadImage(img_path)
        out_path = os.path.join(ARTIFACTS_DIR, f"converted_{os.path.basename(img_path)}")
        write_nifti(img, out_path)
        converted_paths.append(out_path)
    return converted_paths

# Filter (Gaussian smoothing)
@asset
def image_filter(image_conversion: List[str]) -> List[str]:
    filtered_paths = []
    for img_path in image_conversion:
        img = sitk.ReadImage(img_path)
        filtered_img = sitk.SmoothingRecursiveGaussian(img, sigma=1.0)
        out_path = os.path.join(ARTIFACTS_DIR, f"filtered_{os.path.basename(img_path)}")
        write_nifti(filtered_img, out_path)
        filtered_paths.append(out_path)
    return filtered_paths

# Mask registration (nearest neighbor to filtered image geometry)
@asset
def mask_registration(image_filter: List[str], all_masks: List[str]) -> List[str]:
    if not image_filter or not all_masks:
        raise FileNotFoundError("Missing filtered images or masks for mask_registration.")

    registered_mask_paths = []

    # Pair masks to images if lengths match; otherwise, resample all masks to the first filtered image
    if len(image_filter) == len(all_masks):
        pairs = zip(image_filter, all_masks)
    else:
        ref_path = image_filter[0]
        pairs = [(ref_path, m) for m in all_masks]

    for ref_img_path, mask_path in pairs:
        ref_img = sitk.ReadImage(ref_img_path)
        mask_img = sitk.ReadImage(mask_path)

        identity = sitk.Transform(ref_img.GetDimension(), sitk.sitkIdentity)
        resampled_mask = sitk.Resample(
            mask_img, ref_img, identity, sitk.sitkNearestNeighbor, 0, mask_img.GetPixelID()
        )

        out_path = os.path.join(ARTIFACTS_DIR, f"mask_registered_{os.path.basename(mask_path)}")
        sitk.WriteImage(resampled_mask, out_path)
        registered_mask_paths.append(out_path)

    return registered_mask_paths

# Feature extraction (PySeRA, returns JSON-serializable summary)

@asset
def feature_extraction(image_filter: List[str], mask_registration: List[str]) -> Dict[str, list]:
    results = []
    for img, mask in zip(image_filter, mask_registration):
        start = time.time()

        result = pysera.process_batch(
            image_input=img,
            mask_input=mask,
            output_path=ARTIFACTS_DIR,
            categories="diag,morph,glcm,glrlm,glszm,ngtdm,ngldm",
            dimensions="1st,3D",
            bin_size=25,
            roi_num=2,
            roi_selection_mode="per_region",
            apply_preprocessing=True,
            feature_value_mode="REAL_VALUE",
            min_roi_volume=50,
            enable_parallelism=True,
            num_workers=4,
            report="info",
            temporary_files_path=r"C:\\Users\\Omen16\\AppData\\Local\\ViSERA\\res\\memory\\memmap\\pysera_temp",
            IBSI_based_parameters={
                "radiomics_DataType": "CT",
                "radiomics_DiscType": "FBS",
                "radiomics_isScale": 0,
                "radiomics_VoxInterp": "Nearest",
                "radiomics_ROIInterp": "Nearest",
                "radiomics_isotVoxSize": 2.0,
                "radiomics_isotVoxSize2D": 2.0,
                "radiomics_isIsot2D": 0,
                "radiomics_isGLround": 0,
                "radiomics_isReSegRng": 0,
                "radiomics_isOutliers": 0,
                "radiomics_isQuntzStat": 1,
                "radiomics_ReSegIntrvl01": -1000,
                "radiomics_ReSegIntrvl02": 400,
                "radiomics_ROI_PV": 0.5,
                "radiomics_qntz": "Uniform",
                "radiomics_IVH_Type": 3,
                "radiomics_IVH_DiscCont": 1,
                "radiomics_IVH_binSize": 2.0,
            },
        )

        elapsed = round(time.time() - start, 2)

        # Persist detailed per-case result for inspection
        safe_result = convert_paths_and_dfs(result)
        case_json = os.path.join(ARTIFACTS_DIR, f"{os.path.basename(img)}_radiomics.json")
        with open(case_json, "w", encoding="utf-8") as f:
            json.dump(safe_result, f, indent=2, ensure_ascii=False)

        print(f"Radiomics for {os.path.basename(img)} completed in {elapsed:.2f} seconds")

        # Append summary record
        results.append({
            "image": img,
            "mask": mask,
            "elapsed_seconds": elapsed,
            "result_file": case_json,
        })

    return {"radiomics_results": results}

# Final writer (summary JSON)

@asset
def image_write(feature_extraction: Dict[str, list]) -> str:
    out_path = os.path.join(ARTIFACTS_DIR, "final_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feature_extraction, f, indent=2, ensure_ascii=False)
    return out_path
