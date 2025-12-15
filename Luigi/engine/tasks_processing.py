import luigi
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from engine.tasks_reader import ImageReader
from engine.utils import ensure_dir

class ImageFusion(luigi.Task):
    image_file = luigi.Parameter()
    mask_file = luigi.Parameter(default="")
    workspace = luigi.Parameter(default="artifacts")

    def requires(self):
        return ImageReader(image_file=self.image_file, mask_file=self.mask_file, workspace=self.workspace)

    def output(self):
        out_dir = ensure_dir(Path(self.workspace) / "pipeline")
        stem = Path(self.image_file).stem
        return luigi.LocalTarget(str(out_dir / f"fusion_{stem}.json"))

    def run(self):
        # If we had the second modality, we would read here and stack the channels.
        # For now, we're passing that single image along with the metadata.
        payload = {"status": "fused", "modalities": 1, "image": str(self.image_file)}
        with self.output().open("w") as f:
            json.dump(payload, f, indent=2)


class ImageConversion(luigi.Task):
    image_file = luigi.Parameter()
    mask_file = luigi.Parameter(default="")
    workspace = luigi.Parameter(default="artifacts")

    def requires(self):
        return ImageFusion(image_file=self.image_file, mask_file=self.mask_file, workspace=self.workspace)

    def output(self):
        out_dir = ensure_dir(Path(self.workspace) / "pipeline")
        stem = Path(self.image_file).stem
        return luigi.LocalTarget(str(out_dir / f"conversion_{stem}.json"))

    def run(self):
        # Convert to SimpleITK image for later steps
        sitk_img = sitk.ReadImage(str(self.image_file))
        # Type conversion/normalization
        payload = {"status": "converted", "pixel_type": str(sitk_img.GetPixelIDTypeAsString())}
        with self.output().open("w") as f:
            json.dump(payload, f, indent=2)


class ImageFilter(luigi.Task):
    image_file = luigi.Parameter()
    mask_file = luigi.Parameter(default="")
    workspace = luigi.Parameter(default="artifacts")
    sigma = luigi.FloatParameter(default=1.0)

    def requires(self):
        return ImageConversion(image_file=self.image_file, mask_file=self.mask_file, workspace=self.workspace)

    def output(self):
        out_dir = ensure_dir(Path(self.workspace) / "pipeline")
        stem = Path(self.image_file).stem
        return luigi.LocalTarget(str(out_dir / f"filter_{stem}.json"))

    def run(self):
        img = sitk.ReadImage(str(self.image_file))
        # Gaussian smoothing for ROI preparation
        filtered = sitk.DiscreteGaussian(img, variance=self.sigma ** 2)
        # Store Data
        payload = {"status": "filtered", "sigma": self.sigma}
        with self.output().open("w") as f:
            json.dump(payload, f, indent=2)


class MaskRegistration(luigi.Task):
    image_file = luigi.Parameter()
    mask_file = luigi.Parameter(default="")
    workspace = luigi.Parameter(default="artifacts")

    def requires(self):
        return ImageFilter(image_file=self.image_file, mask_file=self.mask_file, workspace=self.workspace)

    def output(self):
        out_dir = ensure_dir(Path(self.workspace) / "pipeline")
        stem = Path(self.image_file).stem
        return luigi.LocalTarget(str(out_dir / f"maskreg_{stem}.json"))

    def run(self):
        # If we have a mask, we register/resample to image space.
        result = {"status": "mask_registered", "mask_available": bool(self.mask_file)}
        if self.mask_file:
            img = sitk.ReadImage(str(self.image_file))
            msk = sitk.ReadImage(str(self.mask_file))
            # Resample mask to image geometry
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            msk_res = resampler.Execute(msk)
            # Temporary storage of PySera results
            tmp_dir = ensure_dir(Path(self.workspace) / "tmp")
            out_mask = Path(tmp_dir) / f"regmask_{Path(self.image_file).stem}.nii.gz"
            sitk.WriteImage(msk_res, str(out_mask))
            result["registered_mask_path"] = str(out_mask)
        with self.output().open("w") as f:
            json.dump(result, f, indent=2)
