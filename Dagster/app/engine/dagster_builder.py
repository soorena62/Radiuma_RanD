from dagster import op, job, In
import os, json
from app.features.pysera_node import run_pysera

DATA_DIR = os.path.join("data", "images")
MASK_DIR = os.path.join("data", "masks")
OUTPUT_DIR = os.path.join("app", "storage", "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@op(ins={"filename": In(str)})
def load_image(filename: str) -> str:
    # Return normalized image file path (do not load arrays/objects)
    return os.path.normpath(os.path.join(DATA_DIR, filename))

@op(ins={"filename": In(str)})
def load_mask(filename: str) -> str:
    # Return normalized mask file path (do not load arrays/objects)
    return os.path.normpath(os.path.join(MASK_DIR, filename))

@op
def extract_features(img_path: str, mask_path: str) -> str:
    # Validate file paths and call PySERA
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image path not found: {img_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask path not found: {mask_path}")

    features = run_pysera(img_path, mask_path)
    out_path = os.path.join(OUTPUT_DIR, "CT_pitch_features.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    return out_path

@job
def radiuma_job():
    # Compose ops: produce file paths, then extract features
    img = load_image()
    mask = load_mask()
    extract_features(img, mask)
