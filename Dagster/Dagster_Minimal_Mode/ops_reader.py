# ops_reader.py
from dagster import op
import os
import pysera

@op
def read_images():
    # Match Radiuma.exe by processing a single pair or a flat folder.
    # Here we keep folders; PySERA will find matching pairs.
    image_dir = "data/images"
    mask_dir = "data/masks"
    return {"image_dir": image_dir, "mask_dir": mask_dir}

@op
def extract_features(data):
    # Exact PySERA config mirrored from Radiuma.exe logs
    result = pysera.process_batch(
        image_input=data["image_dir"],
        mask_input=data["mask_dir"],
        output_path="./artifacts/results",
        # Core run behavior
        enable_parallelism=False,
        num_workers=1,
        apply_preprocessing=True,
        roi_selection_mode="per_region",
        roi_num=2,
        min_roi_volume=50,
        feature_value_mode="REAL_VALUE",
        # Feature scope
        categories="diag,morph,glcm,glrlm,glszm,ngtdm,ngldm",
        dimensions="1st,2D",
        bin_size=25,
        # Logging/report
        report="info",
        # Temp path (matching Radiuma.exe run)
        temporary_files_path=r"C:\Users\Omen16\AppData\Local\ViSERA\res\memory\memmap\pysera_temp",
        # IBSI-based parameters mirrored from Radiuma.exe
        IBSI_based_parameters={
            "radiomics_DataType": "CT",
            "radiomics_DiscType": "FBS",
            "radiomics_isScale": 0,
            "radiomics_VoxInterp": "Nearest",
            "radiomics_ROIInterp": "Nearest",
            "radiomics_isotVoxSize": 1.0,
            "radiomics_isotVoxSize2D": 2.0,
            "radiomics_isIsot2D": 0,
            "radiomics_isGLround": 0,
            "radiomics_isReSegRng": 0,
            "radiomics_isOutliers": 0,
            "radiomics_isQuntzStat": 1,
            "radiomics_ReSegIntrvl01": -1000,
            "radiomics_ReSegIntrvl02": 400,
            "radiomics_ROI_PV": 0.5,
            "radiomics_qntz": "Uniform",
            "radiomics_IVH_Type": 3,
            "radiomics_IVH_DiscCont": 1,
            "radiomics_IVH_binSize": 2.0,
        },
    )
    return result

@op
def write_report(result):
    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/radiomics_batch_report_radiuma_match.txt"

    with open(report_path, "w") as f:
        f.write(f"Success: {result['success']}\n")
        f.write(f"Processed files: {result['processed_files']}\n")
        f.write(f"Processing time: {result['processing_time']:.2f} seconds\n")
        f.write(f"Output path: {result['output_path']}\n")
        f.write("Note: Excel with Radiomics_Features, Parameters, Report is saved in output_path.\n")

    return report_path