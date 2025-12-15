import os
import luigi
import numpy as np
import SimpleITK as sitk
from pathlib import Path

class ImageFusion(luigi.Task):
    artifacts_dir = luigi.Parameter(default="artifacts")

    def requires(self):
        from engine.tasks_registration import ImageRegistration
        return ImageRegistration(artifacts_dir=self.artifacts_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.artifacts_dir, "fused_index.txt"))

    def run(self):
        reg_index = os.path.join(self.artifacts_dir, "registered_index.txt")
        with open(reg_index, "r") as f:
            reg_paths = [line.strip() for line in f if line.strip()]

        fused_paths = []
        for img_path in reg_paths:
            img = sitk.ReadImage(img_path)
            arr = sitk.GetArrayFromImage(img)
            p5, p95 = np.percentile(arr, [5, 95])
            arr = np.clip(arr, p5, p95)
            arr = (arr - p5) / (p95 - p5) if p95 > p5 else arr * 0.0
            fused_img = sitk.GetImageFromArray(arr)
            fused_img.CopyInformation(img)
            out_path = Path(self.artifacts_dir) / f"fused_{Path(img_path).name}"
            sitk.WriteImage(fused_img, str(out_path))
            fused_paths.append(str(out_path))

        with self.output().open("w") as f:
            f.write("\n".join(fused_paths))
