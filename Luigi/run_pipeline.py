import luigi
import time
from engine.pipeline import RadiumaPipeline

if __name__ == "__main__":
    start = time.time()
    luigi.build([RadiumaPipeline(artifacts_dir="artifacts")], local_scheduler=True)
    elapsed = time.time() - start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    centiseconds = int((seconds - int(seconds)) * 100)
    print(f"[Pipeline] total execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}.{centiseconds:02}")
    print("=== Workflow completed successfully ===")
    