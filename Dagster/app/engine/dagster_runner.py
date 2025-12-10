import os
import pysera
from app.engine import radiuma_assets
from app.engine.dagster_builder import radiuma_job
from dagster import materialize, FilesystemIOManager

# Main workflow runner for Radiuma using PySERA and Dagster

def run_workflow():
    # Define input image and mask paths
    image_path = os.path.normpath("data/images/CT_pitch.nii.gz")
    mask_path  = os.path.normpath("data/masks/CT_pitch_mask.nii.gz")
    output_dir = os.path.normpath("./results")

    # Run PySERA radiomics workflow
    # Changed dimensions from "3d" to "1st,2D"
    result = pysera.process_batch(
        image_input=image_path,
        mask_input=mask_path,
        output_path=output_dir,
        categories="all",
        dimensions="1st,2D",  # <-- only change applied here
        apply_preprocessing=True,
    )
    print("✅ PySERA workflow finished." if result.get("success") else "❌ PySERA workflow failed.")


def run_pipeline():
    # Run Dagster job with sample configuration
    result = radiuma_job.execute_in_process(
        run_config={
            "ops": {
                "load_image": {"inputs": {"filename": {"value": "CT_pitch.nii.gz"}}},
                "load_mask":  {"inputs": {"filename": {"value": "CT_pitch_mask.nii.gz"}}},
            }
        }
    )
    print("✅ Dagster job finished.")
    print(result)


def run_assets():
    # Ensure artifacts directory exists
    base_dir = os.path.normpath("app/storage/artifacts")
    os.makedirs(base_dir, exist_ok=True)

    # Use FilesystemIOManager for asset storage
    io_manager = FilesystemIOManager(base_dir=base_dir)

    # Materialize all assets defined in radiuma_assets
    result = materialize(
        [
            radiuma_assets.raw_image,
            radiuma_assets.raw_mask,
            radiuma_assets.registered_image,
            radiuma_assets.fused_image,
            radiuma_assets.features,
        ],
        resources={"io_manager": io_manager},
    )

    if result.success:
        print("✅ Assets materialized successfully.")
        print(f"Storage base_dir: {base_dir}")

        # Print all materialized asset keys
        for event in result.get_asset_materialization_events():
            print(f"Asset built: {event.asset_key}")
    else:
        print("❌ Asset execution failed.")
        print(result)


if __name__ == "__main__":
    print("🚀 Starting Radiuma unified run...\n")
    run_workflow()
    run_pipeline()
    run_assets()
    print("\n🎯 All workflows completed in one run.")
