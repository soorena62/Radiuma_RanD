import os
import luigi
import SimpleITK as sitk
from pathlib import Path

class MaskRegistration(luigi.Task):
    artifacts_dir = luigi.Parameter(default="artifacts")
    mask_dir = luigi.Parameter(default=os.path.join("data", "masks"))

    def requires(self):
        from engine.tasks_filter import ImageFilter
        from engine.tasks_masks import AllMasks
        return {
            "filter": ImageFilter(artifacts_dir=self.artifacts_dir),
            "masks": AllMasks(mask_dir=self.mask_dir)
        }

    def output(self):
        return luigi.LocalTarget(os.path.join(self.artifacts_dir, "mask_registered_index.txt"))

    def run(self):
        filt_index = os.path.join(self.artifacts_dir, "filtered_index.txt")
        with open(filt_index, "r") as f:
            filtered_paths = [line.strip() for line in f if line.strip()]

        mask_files = [os.path.join(self.mask_dir, f) for f in os.listdir(self.mask_dir) if f.endswith(".nii.gz")]
        if not filtered_paths or not mask_files:
            raise FileNotFoundError("Missing filtered images or masks for mask_registration.")

        if len(filtered_paths) == len(mask_files):
            pairs = zip(filtered_paths, mask_files)
        else:
            ref_path = filtered_paths[0]
            pairs = [(ref_path, m) for m in mask_files]

        out_paths = []
        for ref_img_path, mask_path in pairs:
            ref_img = sitk.ReadImage(ref_img_path)
            mask_img = sitk.ReadImage(mask_path)
            identity = sitk.Transform(ref_img.GetDimension(), sitk.sitkIdentity)
            resampled_mask = sitk.Resample(
                mask_img, ref_img, identity, sitk.sitkNearestNeighbor, 0, mask_img.GetPixelID()
            )
            out_path = Path(self.artifacts_dir) / f"mask_registered_{Path(mask_path).name}"
            sitk.WriteImage(resampled_mask, str(out_path))
            out_paths.append(str(out_path))

        with self.output().open("w") as f:
            f.write("\n".join(out_paths))
