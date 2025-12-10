import time
import luigi
from engine.pipeline import RadiumaPipeline

if __name__ == "__main__":
    start = time.time()

    luigi.build([
        RadiumaPipeline(
            image_path="data/images/64T1.nii.gz",
            roi_path="data/masks/64T1_mask.nii.gz",
            dimensions="3d",
            workspace="artifacts",
            tool_version="0.1.0"
        )
    ], local_scheduler=True, detailed_summary=True)

    end = time.time()
    duration = end - start
    print(f"[Total] Workflow finished in {duration:.2f} seconds")

    with open("artifacts/pipeline/final_report.txt", "a", encoding="utf-8") as f:
        f.write(f"\nTotal execution time: {duration:.2f} seconds\n")
