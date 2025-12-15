import yaml
from dagster_radiuma.jobs import radiomics_job, radiomics_batch_job
import app.gui.visual_editor as visual_editor


def run_radiomics_pipeline(config_file: str):
    # Load YAML config
    with open(config_file, "r", encoding="utf-8") as f:
        run_config = yaml.safe_load(f)

    # Check which job to run
    if "radiomics_batch_job" in run_config:
        print("[INFO] Batch mode detected → Running Batch Workflow")
        result = radiomics_batch_job.execute_in_process(
            run_config=run_config["radiomics_batch_job"]
        )
        print(f"[RESULT] Batch workflow success = {result.success}")

    elif "radiomics_job" in run_config:
        print("[INFO] Single case detected → Running Single Workflow")
        result = radiomics_job.execute_in_process(
            run_config=run_config["radiomics_job"]
        )
        print(f"[RESULT] Single workflow success = {result.success}")

    else:
        raise ValueError(
            "Config file must contain either 'radiomics_job' or 'radiomics_batch_job' section."
        )


if __name__ == "__main__":
    print("[INFO] Radiuma Workflow Editor started.")

    use_gui = True   #

    if use_gui:
        
        visual_editor.main()
    else:

        run_radiomics_pipeline("configs.yaml")
