from typing import Literal, override

from trl.trainer.dpo_trainer import DPOTrainer

from scripts.tracker import Tracker

class DITTOTrainer(DPOTrainer):

    def __init__(self, *args, tracker: Tracker = None, **kwargs):
        self.tracker = tracker
        super().__init__(*args, **kwargs)

    @override
    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
        # Add tracker for metrics
        self.tracker.add_metrics(metrics)
