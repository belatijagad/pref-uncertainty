from typing import Any

from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback

from scripts.collator import DITTOCollator


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold=2.0):
        self.threshold = threshold

    def on_step_begin(self, args, state, control, **kwargs):
        if len(state.log_history) > 0:
            last_loss = float("inf")

            for k in state.log_history[::-1]:
                if "loss" in k:
                    last_loss = k["loss"]
                    break

            if last_loss < self.threshold:
                control.should_training_stop = True


class ResampleCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset | IterableDataset,
        collator: DITTOCollator,
        config: dict[str, Any],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.collator = collator
        self.config = config

        self.last_step_num = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.reset_and_resample(args, state, control, **kwargs)

    def on_step_begin(self, args, state, control, **kwargs):
        self.reset_and_resample(args, state, control, **kwargs)

    def reset_and_resample(self, args, state, control, **kwargs):
        step_num = int(state.global_step)

        if self.last_step_num == step_num:
            return

        if step_num % self.config.get("resample_rate", 50) == 0:
            was_training = self.model.training
            self.model.eval()
            
            self.collator.resample(
                step=step_num,
                model=self.model,
                tokenizer=self.tokenizer,
                dataset=self.dataset,
            )
            
            if was_training:
                self.model.train()

        self.last_step_num = step_num
