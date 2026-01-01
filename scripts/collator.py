# DITTO Authors: Omar Shaikh, Michelle S. Lam, Joey Hejna, Yijia Shao, Hyundong Cho, Michael S. Bernstein, Diyi Yang
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.

import random
from typing import Any
from dataclasses import dataclass, field

import torch
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.utils import pad

from scripts.utils import generate_model_outputs
from scripts.estimator import BaseEstimator
from scripts.tracker import Tracker

@dataclass
class DITTOCollator(DataCollatorForPreference):
    """
    DITTO DataCollator that handles dynamic sampling of generated responses 
    (Expert, Replay, Noisy) and correctly formats them for DPO training.
    """
    frac_expert: float = 0.7
    frac_replay: float = 0.2
    frac_noisy: float = 0.1
    rescale_batch: int = 2
    resample_rate: int = 10
    higher_is_better: bool = False
    rejection_thresh: float = 0.0
    gen_kwargs: dict = field(default_factory=dict)
    tracker: Tracker | None = None

    tokenizer: PreTrainedTokenizerBase = None
    estimator: BaseEstimator = None

    # dict[timestep: int, dict[ prompt: str, list[ outputs: dict[str, torch.tensor] ] ] ]
    cache: dict[int, dict[str, list[ dict[str, torch.Tensor] ]]] = field(default_factory=dict)
    sampled_step: int = field(default=0, init=False)

    def __post_init__(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to `DITTOCollator`.")
        
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.pad_token_id

    def set_mode(self, *, training: bool) -> None:
        self.mode = "train" if training else "eval"

    def resample(
        self,
        step: int,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset | IterableDataset,
    ) -> None:
        """
        Generates new responses from the model and updates the cache.
        """
        self.sampled_step = step
        self.cache.setdefault(step, {})

        logging = {"generations": {}}

        prompts = list(dataset["prompt"])
        
        (
            prompt_input_ids, 
            generated_input_ids, 
            scores_view, 
            logits_view
        ) = generate_model_outputs(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            gen_kwargs=self.gen_kwargs,
        )

        for prompt, pr_ids, gen_ids, scores, logits in zip(
            prompts,
            prompt_input_ids, generated_input_ids,
            scores_view, logits_view, strict=True
        ):
            cache_slot = self.cache[step].setdefault(prompt, [])
            pr_id = pr_ids[0] 
            prompt_str = tokenizer.decode(pr_id)
            results = []
            
            # For each generated sequences for the same prompt
            for gen_id, score, logit in zip(
                gen_ids, scores, logits, strict=True
            ):
                
                cache_slot.append(
                    {
                        "score": self.estimator(gen_id, score, logit),
                        "prompt_input_ids": pr_id,
                        "generated_input_ids": gen_id,
                    }
                )
                generation_str = tokenizer.decode(gen_id)
                results.append(generation_str)

            logging["generations"][prompt_str] = results
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.tracker is not None:
            self.tracker.add_generations(logging)

    def _get_noisy_pairs(
            self, 
            prompt: list[torch.Tensor], 
            step_a: int,
        ) -> list[tuple]:
        """
        Generates 'noisy' preference pairs by comparing scores of different generations.
        """
        noisy_pairs = []

        step_data = self.cache.get(step_a, {})
        if prompt not in step_data:
            return noisy_pairs

        current_entries = self.cache[step_a][prompt]
        prompt_input_ids = current_entries[0]["prompt_input_ids"]
        
        # Look at all previous steps
        for step_b in range(step_a):
            past_entries = self.cache.get(step_b, {}).get(prompt)
            if not past_entries:
                continue

            for current in current_entries:
                for past in past_entries:
                    curr_score, past_score = float(current["score"]), float(past["score"])

                    # Rejection sampling
                    margin = abs(curr_score - past_score)
                    if margin < self.rejection_thresh:
                        continue

                    past_is_better = (
                        (past_score > curr_score) 
                        if self.higher_is_better 
                        else (past_score < curr_score)
                    )
                    chosen = past if past_is_better else current
                    rejected = current if past_is_better else past

                    noisy_pairs.append((
                        prompt_input_ids, 
                        chosen["generated_input_ids"], 
                        rejected["generated_input_ids"],
                        # Uncertainty scores
                        (float(chosen["score"]), float(rejected["score"])),
                    ))

        return noisy_pairs

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        def attn_mask(input_ids: list[torch.Tensor]) -> list[torch.Tensor]:
            return [torch.ones_like(input_id) for input_id in input_ids]
        
        # list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        expert_samples = []
        replay_samples = []
        noisy_samples = []

        for example in examples:
            prompt_text = example["prompt"]
            
            current_step_data = self.cache.get(self.sampled_step, {}).get(prompt_text, [])
            
            # Expert samples (Current Step)
            for rejected in current_step_data:
                expert_samples.append((
                    example["prompt_input_ids"],
                    example["chosen_input_ids"],
                    rejected["generated_input_ids"],
                ))
            
            # Replay & Noisy samples (Previous Steps)
            for step_a in self.cache.keys():
                if step_a < self.sampled_step:
                    past_data = self.cache[step_a].get(prompt_text, [])
                    for rejected in past_data:
                        replay_samples.append((
                            example["prompt_input_ids"],
                            example["chosen_input_ids"],
                            rejected["generated_input_ids"],
                        ))
                
                # Noisy
                noisy_samples.extend(self._get_noisy_pairs(prompt_text, step_a))

        # Lets try to remove this guard
        # if not expert_samples and not replay_samples and not noisy_samples:
        #     samples = [
        #         (
        #             ex["prompt_input_ids"], 
        #             ex["chosen_input_ids"], 
        #             ex.get("rejected_input_ids", ex["chosen_input_ids"]) # Handle missing rejected
        #         ) 
        #         for ex in examples
        #     ]
        # else:
        n_expert = int(len(expert_samples) * self.frac_expert)
        n_replay = int(len(replay_samples) * self.frac_replay)
        n_noisy  = int(len(noisy_samples)  * self.frac_noisy) # NEED TO FIX; make it lazy

        expert_sampled = random.sample(expert_samples, min(len(expert_samples), n_expert))
        replay_sampled = random.sample(replay_samples, min(len(replay_samples), n_replay))
        noisy_sampled = random.sample(noisy_samples,  min(len(noisy_samples),  n_noisy))

        if self.tracker is not None:
            tracker_logging = {
                "sampled_data": [
                    (self.tokenizer.decode(chosen), self.tokenizer.decode(rejected))
                    for _, chosen, rejected, _
                    in noisy_sampled
                ],
                "uncertainty": [
                    (chosen_score, rejected_score) 
                    for _, _, _, (chosen_score, rejected_score) 
                    in noisy_sampled
                ],
                "margin": [
                    abs(chosen_score - rejected_score)
                    for _, _, _, (chosen_score, rejected_score) 
                    in noisy_sampled
                ],
            }
            self.tracker.add_collator_sampling(tracker_logging)
        
        samples = expert_sampled + replay_sampled + noisy_sampled
        
        if not samples and expert_samples:
            samples = expert_samples

        keys = ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids"]
        
        raw_tensors = {} 
        for i, key in enumerate(keys):
            raw_tensors[key] = [torch.as_tensor(sample[i]) for sample in samples]

        batch = {}
        
        for name, ids in raw_tensors.items():
            padding_side = "left" if "prompt" in name else "right"
            
            batch[name] = pad(
                ids, 
                padding_value=self.tokenizer.pad_token_id, 
                padding_side=padding_side
            )
            
            mask_name = f"{name.split('_')[0]}_attention_mask"
            batch[mask_name] = pad(
                attn_mask(ids), 
                padding_value=0, 
                padding_side=padding_side
            )
        
        return batch
