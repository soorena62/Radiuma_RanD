from dagster import job, op, In
import SimpleITK as sitk
from pathlib import Path
from app.features.pysera_node import run_pysera
# Write Your Codes Here:


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

@op(config_schema={"image_file": str})
def image_reader(context):
    # The Image File Is Read From Config.
    image_file = context.op_config["image_file"]
    input_path = Path("data/images") / image_file
    img = sitk.ReadImage(str(input_path))
    out_path = ARTIFACTS_DIR / "image_reader.nii.gz"
    sitk.WriteImage(img, str(out_path))
    return str(out_path)

@op(config_schema={"mask_file": str})
def mask_reader(context):
    mask_file = context.op_config["mask_file"]
    mask_path = Path("data/masks") / mask_file
    mask = sitk.ReadImage(str(mask_path))
    out_path = ARTIFACTS_DIR / "mask_reader.nii.gz"
    sitk.WriteImage(mask, str(out_path))
    return str(out_path)

@op
def image_registration(image_path: str):
    img = sitk.ReadImage(image_path)
    out_path = ARTIFACTS_DIR / "registered.nii.gz"
    sitk.WriteImage(img, str(out_path))
    return str(out_path)

@op
def image_filter(registered_path: str):
    img = sitk.ReadImage(registered_path)
    filtered = sitk.DiscreteGaussian(img, variance=1.5)
    out_path = ARTIFACTS_DIR / "filtered.nii.gz"
    sitk.WriteImage(filtered, str(out_path))
    return str(out_path)

@op
def pysera_extract(filtered_path: str, mask_path: str):
    features = run_pysera(filtered_path, mask_path)
    out_path = ARTIFACTS_DIR / "features.json"
    out_path.write_text(str(features))
    return str(out_path)

@op
def image_writer(filtered_path: str):
    final_path = ARTIFACTS_DIR / "final_output.nii.gz"
    final_path.write_bytes(Path(filtered_path).read_bytes())
    return str(final_path)

@job
def radiuma_job():
    img = image_reader()
    mask = mask_reader()
    reg = image_registration(img)
    filt = image_filter(reg)
    pysera_extract(filt, mask)
    image_writer(filt)
