import sys
from dagster import materialize
import radiuma_assets

# Ensure UTF-8 output for logs
sys.stdout.reconfigure(encoding="utf-8")

def run_assets():
    print("Starting Dagster Assets run...")

    result = materialize(
        [
            radiuma_assets.all_masks,
            radiuma_assets.image_reader,
            radiuma_assets.image_registration,
            radiuma_assets.image_fusion,
            radiuma_assets.image_conversion,
            radiuma_assets.image_filter,
            radiuma_assets.mask_registration,   # align masks to filtered image geometry
            radiuma_assets.feature_extraction,  # consume filtered images + registered masks
            radiuma_assets.image_write,
        ],
        resources={"io_manager": radiuma_assets.json_io_manager},
    )

    print("Workflow completed successfully." if result.success else "Workflow failed.")

if __name__ == "__main__":
    run_assets()
