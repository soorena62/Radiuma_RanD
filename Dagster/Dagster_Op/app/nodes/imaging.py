from pathlib import Path
import numpy as np
import SimpleITK as sitk

ARTIFACTS = Path("app/storage/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def image_reader_fn(state):
    # Decide input file (DICOM series or NIfTI)
    input_image_path = Path("data/images/CT_AVM.nii.gz")  # adapt as needed
    img = sitk.ReadImage(str(input_image_path))
    out_path = ARTIFACTS / "image_reader.nii.gz"
    sitk.WriteImage(img, str(out_path))
    return {"image_path": str(out_path)}

def image_registration_fn(state):
    fixed_path = state.get_outputs("image_reader").get("image_path")
    moving_path = fixed_path  # demo: register image to itself; replace with real moving image
    fixed = sitk.ReadImage(fixed_path)
    moving = sitk.ReadImage(moving_path)

    # Simple rigid registration (demo)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=100
    )
    registration_method.SetInterpolator(sitk.sitkLinear)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed, moving)

    resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    out_path = ARTIFACTS / "registered.nii.gz"
    sitk.WriteImage(resampled, str(out_path))
    return {"registered_path": str(out_path), "transform": "euler3d"}

def image_filter_fn(state):
    reg_path = state.get_outputs("image_registration").get("registered_path")
    img = sitk.ReadImage(reg_path)
    # Example filter: Gaussian smoothing
    filtered = sitk.DiscreteGaussian(img, variance=1.5)
    out_path = ARTIFACTS / "filtered.nii.gz"
    sitk.WriteImage(filtered, str(out_path))
    return {"filtered_path": str(out_path)}

def image_writer_fn(state):
    # Writer here is just saving; already done by previous steps; simulate final export
    filtered_path = state.get_outputs("image_filter").get("filtered_path")
    final_path = ARTIFACTS / "final_output.nii.gz"
    Path(final_path).write_bytes(Path(filtered_path).read_bytes())
    return {"final_output_path": str(final_path)}
