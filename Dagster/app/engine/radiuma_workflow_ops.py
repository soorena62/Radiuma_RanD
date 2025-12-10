import json
import numpy as np
from dagster import op
import SimpleITK as sitk
from pathlib import Path
# Write Your codes Here:


ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)


@op
def image_reader(path: str):
    """Medical image reading (MRI, CT, PET)."""
    img = sitk.ReadImage(path)
    # Save the processed version for display in the GUI (optional)
    sitk.WriteImage(img, str(ARTIFACTS / "processed_image.nii.gz"))
    return img

@op
def image_registration(fixed_img, moving_img):
    """Aligning two medical images."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    reg.SetInterpolator(sitk.sitkLinear)
    transform = reg.Execute(fixed_img, moving_img)
    registered = sitk.Resample(
        moving_img, fixed_img, transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID()
    )
    sitk.WriteImage(registered, str(ARTIFACTS / "registered_image.nii.gz"))
    return registered

@op
def image_fusion(img1, img2):
    """Combining two medical images (simple fusion)."""
    fused = sitk.Cast((img1 + img2) / 2, sitk.sitkFloat32)
    sitk.WriteImage(fused, str(ARTIFACTS / "fused_image.nii.gz"))
    return fused

@op
def image_extraction(img):
    """Extracting simple features from images."""
    arr = sitk.GetArrayFromImage(img)
    features = {
        "mean_intensity": float(np.mean(arr)),
        "std_intensity": float(np.std(arr)),
        "min_intensity": float(np.min(arr)),
        "max_intensity": float(np.max(arr)),
    }
    (ARTIFACTS / "features.json").write_text(json.dumps(features, indent=2))
    return features

@op
def image_writer(img):
    """Save final output."""
    out_path = ARTIFACTS / "output.nii.gz"
    sitk.WriteImage(img, str(out_path))
    return str(out_path)

OPS = {
    "reader": image_reader,
    "registration": image_registration,
    "fusion": image_fusion,
    "extraction": image_extraction,
    "writer": image_writer,
}

