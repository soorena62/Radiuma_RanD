import os
import numpy as np
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from dagster import op, In, Out, DynamicOut, DynamicOutput
# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def remove_ext(filename: str, exts):
    fname = filename
    for ext in exts:
        if fname.lower().endswith(ext.lower()):
            return fname[:-len(ext)]
    return fname

def strip_suffixes(name_no_ext: str, suffixes):
    for s in suffixes:
        if name_no_ext.endswith(s):
            return name_no_ext[:-len(s)]
    return name_no_ext

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n]

def list_files(root: Path, exts):
    files = []
    for p in Path(root).rglob("*"):
        if p.is_file() and any(str(p).lower().endswith(ext.lower()) for ext in exts):
            files.append(p)
    return sorted(files)

# ------------------------------------------------------------
# Single-case readers
# ------------------------------------------------------------

@op(
    config_schema={
        "image_path": str,
        "mask_path": str,
    },
    out={"image": Out(), "mask": Out()}
)
def read_image_and_mask(context):
    image_path = context.op_config["image_path"]
    mask_path  = context.op_config["mask_path"]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    image = sitk.ReadImage(image_path)
    mask  = sitk.ReadImage(mask_path)
    img_arr = sitk.GetArrayFromImage(image)
    msk_arr = sitk.GetArrayFromImage(mask)

    context.log.info(f"[single] image:{img_arr.shape} mask:{msk_arr.shape}")
    return img_arr, msk_arr

@op(
    ins={"case": In()},
    out={"image": Out(), "mask": Out()}
)
def read_image_and_mask_from_case(context, case: dict):
    image_path = case["image_path"]
    mask_path  = case["mask_path"]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    image = sitk.ReadImage(image_path)
    mask  = sitk.ReadImage(mask_path)
    img_arr = sitk.GetArrayFromImage(image)
    msk_arr = sitk.GetArrayFromImage(mask)

    context.log.info(f"[case] image:{img_arr.shape} mask:{msk_arr.shape}")
    return img_arr, msk_arr

# ------------------------------------------------------------
# Preprocess, features, writer
# ------------------------------------------------------------

@op(out=Out())
def filter_image(context, image: np.ndarray):
    context.log.info(f"[filter] image:{image.shape}")
    return image

@op(ins={"image": In(), "mask": In()}, out=Out())
def extract_features(context, image: np.ndarray, mask: np.ndarray):
    masked = image[mask > 0]
    mean_val = float(masked.mean()) if masked.size > 0 else 0.0
    features = {"mean_intensity": mean_val}
    context.log.info(f"[features] {features}")
    return features


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

@op(ins={"features": In(), "case_name": In()}, out=Out())
def write_outputs(context, features: dict, case_name: str):
    # Define output CSV file path
    csv_path = RESULTS_DIR / "features.csv"

    # Prepare row with case name and features
    row = {"case_name": case_name}
    row.update(features)

    # If CSV does not exist, create new file with header
    if not csv_path.exists():
        df = pd.DataFrame([row])
        df.to_csv(csv_path, index=False)
    else:
        # Append new row to existing CSV
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode="a", header=False, index=False)

    context.log.info(f"[write] features appended to {csv_path}")

    # Optional Excel export (currently disabled)
    # excel_path = RESULTS_DIR / "features.xlsx"
    # df.to_excel(excel_path, index=False)

    return str(csv_path)

# ------------------------------------------------------------
# Batch case discovery
# ------------------------------------------------------------

@op(
    config_schema={
        "images_dir": str,
        "masks_dir": str,
        "image_exts": list,
        "mask_exts": list,
        "mask_suffixes": list,
        "allow_fuzzy_match": bool,
        "fuzzy_max_distance": int,
        "verbose": bool,
    },
    out=DynamicOut()
)
def enumerate_cases_auto(context):
    cfg = context.op_config
    images_dir = Path(cfg["images_dir"]).expanduser().resolve()
    masks_dir  = Path(cfg["masks_dir"]).expanduser().resolve()
    image_exts = [ext.lower() for ext in cfg["image_exts"]]
    mask_exts  = [ext.lower() for ext in cfg["mask_exts"]]
    mask_suffixes = cfg["mask_suffixes"]
    allow_fuzzy = cfg["allow_fuzzy_match"]
    max_dist    = int(cfg["fuzzy_max_distance"])
    verbose     = bool(cfg.get("verbose", True))

    img_files = list_files(images_dir, image_exts)
    msk_files = list_files(masks_dir, mask_exts)

    context.log.info(f"[discover] images_dir={images_dir} masks_dir={masks_dir}")
    context.log.info(f"[discover] found images: {len(img_files)}")
    for p in img_files:
        context.log.info(f"  - image: {p.name}")
    context.log.info(f"[discover] found masks: {len(msk_files)}")
    for p in msk_files:
        context.log.info(f"  - mask:  {p.name}")

    msk_index = {}
    for m in msk_files:
        base = strip_suffixes(remove_ext(m.name, mask_exts), mask_suffixes).lower()
        msk_index.setdefault(base, []).append(m)

    yielded = 0
    skipped = 0

    for img in img_files:
        base = remove_ext(img.name, image_exts).lower()
        candidates = msk_index.get(base, [])
        chosen = None
        reason = None

        if candidates:
            chosen = candidates[0]
        else:
            if allow_fuzzy and msk_files:
                best = None
                best_d = 10**9
                for m in msk_files:
                    mbase = strip_suffixes(remove_ext(m.name, mask_exts), mask_suffixes).lower()
                    d = levenshtein(base, mbase)
                    if d < best_d:
                        best, best_d = m, d
                if best is not None and best_d <= max_dist:
                    chosen = best
                else:
                    reason = f"no exact or fuzzy match (best distance {best_d})"

        if chosen is None:
            skipped += 1
            context.log.warning(f"[skip] image:{img.name} → no mask found ({reason}) | base(image)='{base}'")
            continue

        case = {
            "image_path": str(img),
            "mask_path":  str(chosen),
            "case_name":  remove_ext(img.name, image_exts),
        }
        context.log.info(f"[pair] {case['case_name']} → image:{img.name} mask:{Path(chosen).name}")
        yield DynamicOutput(case, mapping_key=case["case_name"])
        yielded += 1

    context.log.info(f"[summary] yielded={yielded} skipped={skipped}")

    if yielded == 0:
        context.log.error("No valid cases discovered. See [skip] messages above for reasons.")
        raise Exception("No valid cases discovered. See logs for details.")
