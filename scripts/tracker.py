import json
from pathlib import Path

class Tracker:
    def __init__(self, run_dir: str | Path = "logs"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.internal_step = 0
        self.gen_file = self.run_dir / "generations.jsonl"
        self.sample_file = self.run_dir / "samples.jsonl"

        self.generations = []
        self.samples = []

    def add_generations(self, _generations: dict[str, any]):
        generations = _generations.copy()
        generations["step"] = self.internal_step
        self.generations.append(generations)

    def add_collator_sampling(self, _samples: dict[str, any]):
        samples = _samples.copy()
        samples["step"] = self.internal_step
        
        self.samples.append(samples)

    def add_metrics(self, metrics: dict[str, any]):
        if self.internal_step < len(self.samples):
            self.samples[self.internal_step]["metrics"] = metrics
        
        self.internal_step += 1

    def save(self):
        print(f"Saving {len(self.generations)} generations to {self.gen_file}...")
        with self.gen_file.open("w") as f:
            for item in self.generations:
                f.write(json.dumps(item) + "\n")

        print(f"Saving {len(self.samples)} samples to {self.sample_file}...")
        with self.sample_file.open("w") as f:
            for item in self.samples:
                f.write(json.dumps(item) + "\n")
                