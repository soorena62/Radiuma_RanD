# app/gui/napari_main.py
import napari
from pathlib import Path
import SimpleITK as sitk
import numpy as np

def sitk_to_numpy(img):
    return sitk.GetArrayFromImage(img)  # z, y, x

def run_napari_viewer():
    viewer = napari.Viewer(title="Radiuma Mini - Viewer")
    # Load outputs if they exist
    filtered = Path("app/storage/artifacts/filtered.nii.gz")
    mask = Path("data/masks/CT_AVM_mask.nii.gz")
    if filtered.exists():
        arr = sitk_to_numpy(sitk.ReadImage(str(filtered)))
        viewer.add_image(arr, name="filtered", blending="translucent")
    if mask.exists():
        marr = sitk_to_numpy(sitk.ReadImage(str(mask)))
        viewer.add_labels(marr, name="mask")
    napari.run()

if __name__ == "__main__":
    run_napari_viewer()
