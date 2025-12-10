import os, hashlib, luigi
import pysera
from engine.tasks_reader import ImageReaderTask
from engine.provenance import write_sidecar

class FeatureExtractionTask(luigi.Task):
    image_path   = luigi.Parameter()
    roi_path     = luigi.Parameter(default="")
    dimensions   = luigi.Parameter(default="3d")
    workspace    = luigi.Parameter(default="artifacts")
    tool_version = luigi.Parameter(default="0.1.0")

    def requires(self):
        # This task depends on ImageReaderTask
        return ImageReaderTask(
            self.image_path,
            workspace=self.workspace,
            tool_version=self.tool_version
        )

    def output(self):
        # Build output directory based on hash of image+mask path
        out_dir = os.path.join(
            self.workspace,
            f"radiomics{self.dimensions}",
            hashlib.md5((self.image_path + self.roi_path).encode()).hexdigest()
        )
        os.makedirs(out_dir, exist_ok=True)
        return [
            luigi.LocalTarget(os.path.join(out_dir, "features.csv")),
            luigi.LocalTarget(os.path.join(out_dir, "report.md"))
        ]

    def run(self):
        # Call PySERA to process the image + mask
        result = pysera.process_batch(
            image_input=self.image_path,
            mask_input=self.roi_path,
            output_path=os.path.dirname(self.output()[0].path)
        )

        # Save features into CSV
        with self.output()[0].open("w") as f:
            f.write("feature,value\n")
            for k, v in result.items():
                f.write(f"{k},{v}\n")

        # Save a simple report
        with self.output()[1].open("w") as f:
            f.write(f"# Radiomics {self.dimensions.upper()} Report\n")
            f.write(f"Extracted {len(result)} features from {self.image_path}\n")

        # Provenance metadata
        write_sidecar(self.output(), {
            "stage": "feature_extraction",
            "radiomics_dimensions": self.dimensions,
            "image_path": self.image_path,
            "roi_path": self.roi_path,
            "tool_version": self.tool_version
        })
