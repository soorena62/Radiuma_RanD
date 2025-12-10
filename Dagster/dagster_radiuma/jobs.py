from dagster import job, graph, op
from pathlib import Path
from .ops import (
    read_image_and_mask,
    read_image_and_mask_from_case,
    filter_image,
    extract_features,
    write_outputs,
    enumerate_cases_auto
)

# استخراج نام فایل تصویر از config برای حالت تک‌فایلی
@op(config_schema={"image_path": str})
def get_case_name_from_config(context) -> str:
    image_path = context.op_config["image_path"]
    return Path(image_path).stem

# استخراج case_name از دیکشنری case برای حالت batch
@op
def get_case_name_from_case(context, case: dict) -> str:
    return case["case_name"]

# Single-case workflow job
@job(config={
    "ops": {
        "read_image_and_mask": {
            "config": {
                "image_path": "C:/Users/Omen16/Documents/Radiuma_Mini/data/images/CT_pitch.nii.gz",
                "mask_path":  "C:/Users/Omen16/Documents/Radiuma_Mini/data/masks/CT_pitch_mask.nii.gz",
            }
        },
        "get_case_name_from_config": {
            "config": {
                "image_path": "C:/Users/Omen16/Documents/Radiuma_Mini/data/images/CT_pitch.nii.gz"
            }
        }
    }
})
def radiomics_job():
    image, mask = read_image_and_mask()
    filtered = filter_image(image)
    feats = extract_features(image=filtered, mask=mask)
    case_name = get_case_name_from_config()
    write_outputs(features=feats, case_name=case_name)

# Graph that processes one case dict (used for batch mapping)
@graph
def process_case(case):
    image, mask = read_image_and_mask_from_case(case)
    filtered = filter_image(image)
    feats = extract_features(image=filtered, mask=mask)
    case_name = get_case_name_from_case(case)
    write_outputs(features=feats, case_name=case_name)

# Batch workflow job (auto-discovery + dynamic mapping)
@job
def radiomics_batch_job():
    cases = enumerate_cases_auto()
    cases.map(process_case)

# Preprocessing-only workflow job
@job(config={
    "ops": {
        "read_image_and_mask": {
            "config": {
                "image_path": "C:/Users/Omen16/Documents/Radiuma_Mini/data/images/CT_pitch.nii.gz",
                "mask_path":  "C:/Users/Omen16/Documents/Radiuma_Mini/data/masks/CT_pitch_mask.nii.gz",
            }
        },
        "get_case_name_from_config": {
            "config": {
                "image_path": "C:/Users/Omen16/Documents/Radiuma_Mini/data/images/CT_pitch.nii.gz"
            }
        }
    }
})
def preprocessing_job():
    image, mask = read_image_and_mask()
    filtered = filter_image(image)
    feats = extract_features(image=filtered, mask=mask)
    case_name = get_case_name_from_config()
    write_outputs(features=feats, case_name=case_name)

# Feature-extraction-only workflow job
@job(config={
    "ops": {
        "read_image_and_mask": {
            "config": {
                "image_path": "C:/Users/Omen16/Documents/Radiuma_Mini/data/images/CT_pitch.nii.gz",
                "mask_path":  "C:/Users/Omen16/Documents/Radiuma_Mini/data/masks/CT_pitch_mask.nii.gz",
            }
        },
        "get_case_name_from_config": {
            "config": {
                "image_path": "C:/Users/Omen16/Documents/Radiuma_Mini/data/images/CT_pitch.nii.gz"
            }
        }
    }
})
def feature_extraction_job():
    image, mask = read_image_and_mask()
    feats = extract_features(image=image, mask=mask)
    case_name = get_case_name_from_config()
    write_outputs(features=feats, case_name=case_name)