import os
import luigi
import SimpleITK as sitk
from pathlib import Path

class ImageConversion(luigi.Task):
    artifacts_dir = luigi.Parameter(default="artifacts")

    def requires(self):
        from engine.tasks_fusion import ImageFusion
        return ImageFusion(artifacts_dir=self.artifacts_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.artifacts_dir, "converted_index.txt"))

    def run(self):
        fused_index = os.path.join(self.artifacts_dir, "fused_index.txt")
        with open(fused_index, "r") as f:
            fused_paths = [line.strip() for line in f if line.strip()]

        converted_paths = []
        for img_path in fused_paths:
            img = sitk.ReadImage(img_path)
            out_path = Path(self.artifacts_dir) / f"converted_{Path(img_path).name}"
            sitk.WriteImage(img, str(out_path))
            converted_paths.append(str(out_path))

        with self.output().open("w") as f:
            f.write("\n".join(converted_paths))
