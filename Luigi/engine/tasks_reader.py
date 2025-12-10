import os, hashlib, time, luigi
import pydicom, nibabel as nib
import SimpleITK as sitk
from PIL import Image
import numpy as np
from engine.provenance import write_sidecar
from engine.flags import is_set

class ImageReaderTask(luigi.Task):
    image_file   = luigi.Parameter()
    mask_file    = luigi.Parameter(default="")
    workspace    = luigi.Parameter(default="artifacts")
    tool_version = luigi.Parameter(default="0.1.0")   # ✅ added

    def output(self):
        # Build output directory based on hash of the input file path
        out_dir = os.path.join(
            self.workspace,
            "reader",
            hashlib.md5(self.image_file.encode()).hexdigest()
        )
        os.makedirs(out_dir, exist_ok=True)
        return luigi.LocalTarget(os.path.join(out_dir, "image.npy"))

    def run(self):
        # Detect format by full filename, not just last extension
        ext = self.image_file.lower()
        if ext.endswith(".dcm"):
            ds = pydicom.dcmread(self.image_file)
            arr = ds.pixel_array
        elif ext.endswith(".nii") or ext.endswith(".nii.gz"):
            img = nib.load(self.image_file)
            arr = img.get_fdata()
        elif ext.endswith(".png") or ext.endswith(".jpg") or ext.endswith(".jpeg") or ext.endswith(".tif"):
            arr = np.array(Image.open(self.image_file))
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Save as numpy array for later workflow steps
        np.save(self.output().path, arr)

        # Write provenance sidecar file
        write_sidecar([self.output()], {
            "stage": "reader",
            "image_file": self.image_file,
            "mask_file": self.mask_file,
            "tool_version": self.tool_version
        })