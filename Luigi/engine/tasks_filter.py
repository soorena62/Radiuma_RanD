import os
import luigi
import SimpleITK as sitk
from pathlib import Path

class ImageFilter(luigi.Task):
    artifacts_dir = luigi.Parameter(default="artifacts")
    sigma = luigi.FloatParameter(default=1.0)

    def requires(self):
        from engine.tasks_conversion import ImageConversion
        return ImageConversion(artifacts_dir=self.artifacts_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.artifacts_dir, "filtered_index.txt"))

    def run(self):
        conv_index = os.path.join(self.artifacts_dir, "converted_index.txt")
        with open(conv_index, "r") as f:
            conv_paths = [line.strip() for line in f if line.strip()]

        filtered_paths = []
        for img_path in conv_paths:
            img = sitk.ReadImage(img_path)
            filtered_img = sitk.SmoothingRecursiveGaussian(img, sigma=self.sigma)
            out_path = Path(self.artifacts_dir) / f"filtered_{Path(img_path).name}"
            sitk.WriteImage(filtered_img, str(out_path))
            filtered_paths.append(str(out_path))

        with self.output().open("w") as f:
            f.write("\n".join(filtered_paths))
