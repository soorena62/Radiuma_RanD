import luigi
import time
from engine.tasks_writer import ImageWriter

class RadiumaPipeline(luigi.WrapperTask):
    artifacts_dir = luigi.Parameter(default="artifacts")

    def requires(self):
        # start timer at the very beginning
        self.start_time = time.time()
        return ImageWriter(artifacts_dir=self.artifacts_dir)

    def run(self):
        elapsed = time.time() - self.start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        centiseconds = int((seconds - int(seconds)) * 100)
        print(f"[Pipeline] total execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}.{centiseconds:02}")
        print("=== Workflow completed successfully ===")
