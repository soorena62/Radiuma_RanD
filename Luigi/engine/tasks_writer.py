import os, hashlib, luigi
from engine.tasks_radiomics import FeatureExtractionTask
from engine.provenance import write_sidecar

class WriterTask(luigi.Task):
    image_path = luigi.Parameter()
    roi_path   = luigi.Parameter(default="")
    dimensions = luigi.Parameter(default="3d")
    workspace  = luigi.Parameter(default="artifacts")
    tool_version = luigi.Parameter(default="0.1.0")

    def requires(self):
        return FeatureExtractionTask(
            image_path=self.image_path,
            roi_path=self.roi_path,
            dimensions=self.dimensions,
            workspace=self.workspace,
            tool_version=self.tool_version
        )

    def output(self):
        out_dir = os.path.join(
            self.workspace,
            "pipeline",
            hashlib.md5((self.image_path+self.roi_path+self.dimensions).encode()).hexdigest()
        )
        os.makedirs(out_dir, exist_ok=True)
        return luigi.LocalTarget(os.path.join(out_dir, "final_report.txt"))

    def run(self):
        feat_file, report_file = self.input()
        with feat_file.open("r") as f: features = f.read()
        with report_file.open("r") as f: report = f.read()
        with self.output().open("w") as f:
            f.write("=== Final Radiomics Report ===\n")
            f.write(features + "\n")
            f.write(report + "\n")
        write_sidecar([self.output()], {
            "stage": "writer",
            "image_path": self.image_path,
            "roi_path": self.roi_path,
            "radiomics_dimensions": self.dimensions,
            "tool_version": self.tool_version
        })
