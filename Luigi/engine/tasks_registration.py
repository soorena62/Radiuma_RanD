import os
import luigi
import SimpleITK as sitk
from pathlib import Path
from engine.utils import ensure_dir

def cast_to_float32(img: sitk.Image) -> sitk.Image:
    return sitk.Cast(img, sitk.sitkFloat32)

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

class ImageRegistration(luigi.Task):
    artifacts_dir = luigi.Parameter(default="artifacts")

    def requires(self):
        from engine.tasks_reader import ImageReader
        return ImageReader(artifacts_dir=self.artifacts_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.artifacts_dir, "registered_index.txt"))

    def run(self):
        # Reader Path Reader
        reader_index = os.path.join(self.artifacts_dir, "reader", "reader_index.txt")
        with open(reader_index, "r") as f:
            reader_paths = [line.strip() for line in f if line.strip()]

        fixed_raw = sitk.ReadImage(reader_paths[0])
        fixed = cast_to_float32(fixed_raw)

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
        for img_path in reader_paths:
            moving_raw = sitk.ReadImage(img_path)
            moving = cast_to_float32(moving_raw)
            init_tx = make_initial_transform(fixed, moving)
            R.SetInitialTransform(init_tx, inPlace=False)
            final_tx = R.Execute(fixed, moving)
            registered = sitk.Resample(moving, fixed, final_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            out_path = Path(self.artifacts_dir) / f"registered_{Path(img_path).name}"
            sitk.WriteImage(registered, str(out_path))
            out_paths.append(str(out_path))

        with self.output().open("w") as f:
            f.write("\n".join(out_paths))
