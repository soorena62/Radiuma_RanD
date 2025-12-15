import os
import time
import json
import luigi
from pathlib import Path
import pysera
from engine.utils import json_safe

class FeatureExtraction(luigi.Task):
    artifacts_dir = luigi.Parameter(default="artifacts")
    temp_dir = luigi.Parameter(default=r"C:\Users\Omen16\AppData\Local\ViSERA\res\memory\memmap\pysera_temp")

    def requires(self):
        from engine.tasks_filter import ImageFilter
        from engine.tasks_maskreg import MaskRegistration
        return {
            "filter": ImageFilter(artifacts_dir=self.artifacts_dir),
            "maskreg": MaskRegistration(artifacts_dir=self.artifacts_dir)
        }

    def output(self):
        return luigi.LocalTarget(os.path.join(self.artifacts_dir, "radiomics_index.json"))

    def run(self):
        # 
        filt_index = os.path.join(self.artifacts_dir, "filtered_index.txt")
        mask_index = os.path.join(self.artifacts_dir, "mask_registered_index.txt")
        with open(filt_index, "r") as f:
            filtered_paths = [line.strip() for line in f if line.strip()]
        with open(mask_index, "r") as f:
            mask_paths = [line.strip() for line in f if line.strip()]

        results = []
        for img, mask in zip(filtered_paths, mask_paths):
            start = time.time()
            result = pysera.process_batch(
                image_input=img,
                mask_input=mask,
                output_path=self.artifacts_dir,
                categories="diag,morph,glcm,glrlm,glszm,ngtdm,ngldm",
                dimensions="1st,3D",
                bin_size=25,
                roi_num=2,
                roi_selection_mode="per_region",
                apply_preprocessing=True,
                feature_value_mode="REAL_VALUE",
                min_roi_volume=50,
                enable_parallelism=True,
                num_workers="4",
                report="info",
                temporary_files_path=str(self.temp_dir),
                IBSI_based_parameters={
                    "radiomics_DataType": "CT",
                    "radiomics_DiscType": "FBS",
                    "radiomics_isScale": 0,
                    "radiomics_VoxInterp": "Nearest",
                    "radiomics_ROIInterp": "Nearest",
                    "radiomics_isotVoxSize": 2.0,
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
            elapsed = round(time.time() - start, 2)

            safe_result = json_safe(result)
            case_json = os.path.join(self.artifacts_dir, f"{Path(img).name}_radiomics.json")
            with open(case_json, "w", encoding="utf-8") as f:
                json.dump(safe_result, f, indent=2, ensure_ascii=False)

            results.append({
                "image": img,
                "mask": mask,
                "elapsed_seconds": elapsed,
                "result_file": case_json,
            })

        with self.output().open("w") as f:
            json.dump({"radiomics_results": results}, f, indent=2, ensure_ascii=False)
