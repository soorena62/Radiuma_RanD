# radiuma_pipeline.py
from dagster import job
from ops_reader import read_images, extract_features, write_report

@job
def radiuma_job():
    result = extract_features(read_images())
    write_report(result)