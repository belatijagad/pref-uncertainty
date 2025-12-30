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
        # Each step will be like this
        # {
        #     "step": 0,
        #     "generations": {
        #         "prompt_1": ["gen_1", ...],
        #         "prompt_2": ["gen_2", ...],
        #     }
        # }
        generations = _generations.copy()
        generations["step"] = self.internal_step
        self.generations.extend(generations)

    def add_collator_sampling(self, _samples: dict[str, any]):
        # Each step will be like this
        # {
        #     "step": 1,
        #     "sampled_data": [0, 1],
        #     "uncertainty": [[0.5, 0.7], [0.2, 0.3]],
        #     "margin": [0.2, 0.1],
        #     
        #     # concatenated in the add_metrics
        #     "metrics": {
        #         "what": "ever"
        #     }
        # }
        samples = _samples.copy()
        samples["step"] = self.internal_step
        self.samples.append(samples)
        self.samples[self.internal_step] = samples
        self.samples[self.internal_step]["step"] = self.internal_step

    def add_metrics(self, metrics: dict[str, any]):
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
