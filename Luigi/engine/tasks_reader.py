import os
import luigi
import SimpleITK as sitk
from pathlib import Path
from engine.utils import ensure_dir

class ImageReader(luigi.Task):
    data_dir = luigi.Parameter(default=os.path.join("data", "images"))
    artifacts_dir = luigi.Parameter(default="artifacts")

    def output(self):
        out_dir = ensure_dir(Path(self.artifacts_dir) / "reader")
        return luigi.LocalTarget(str(out_dir / "reader_index.txt"))

    def run(self):
        reader_dir = ensure_dir(Path(self.artifacts_dir) / "reader")
        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".nii.gz")]
        if not files:
            raise FileNotFoundError("No images found in data/images")
        converted_paths = []
        for path in files:
            img = sitk.ReadImage(path)
            img_float = sitk.Cast(img, sitk.sitkFloat32)
            out_path = reader_dir / f"reader_{Path(path).name}"
            sitk.WriteImage(img_float, str(out_path))
            converted_paths.append(str(out_path))
        with self.output().open("w") as f:
            f.write("\n".join(converted_paths))
