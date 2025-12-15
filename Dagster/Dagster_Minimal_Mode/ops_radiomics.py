from dagster import op
import pysera

@op
def extract_features(inputs):
    result = pysera.process_batch(
        image_input="data/images/ourT1.nii.gz",
        mask_input="data/masks/ourT1_mask.nii.gz",
        output_path="artifacts"
    )
    return result
