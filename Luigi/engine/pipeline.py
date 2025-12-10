import luigi
from engine.tasks_writer import WriterTask

class RadiumaPipeline(luigi.Task):
    image_path = luigi.Parameter()
    roi_path   = luigi.Parameter(default="")
    dimensions = luigi.Parameter(default="3d")
    workspace  = luigi.Parameter(default="artifacts")
    tool_version = luigi.Parameter(default="0.1.0")

    def requires(self):
        return WriterTask(image_path=self.image_path, roi_path=self.roi_path,
                          dimensions=self.dimensions, workspace=self.workspace,
                          tool_version=self.tool_version)

    def output(self):
        return self.input()  # The output of WriterTask is the same as the output of Pipeline

    def run(self):
        pass
