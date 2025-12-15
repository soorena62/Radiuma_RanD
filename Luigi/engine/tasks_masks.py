import os
import luigi

class AllMasks(luigi.Task):
    mask_dir = luigi.Parameter(default=os.path.join("data", "masks"))

    def output(self):
        # Merely as a signal of completion
        return luigi.LocalTarget(os.path.join("artifacts", "all_masks.done"))

    def run(self):
        files = [os.path.join(self.mask_dir, f) for f in os.listdir(self.mask_dir) if f.endswith(".nii.gz")]
        if not files:
            raise FileNotFoundError("No masks found in data/masks")
        # Just make the done signal.
        with self.output().open("w") as f:
            f.write(f"{len(files)} masks discovered")
