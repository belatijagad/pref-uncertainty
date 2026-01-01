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

from scripts.utils import generate_model_outputs, format_response
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

        # Extract formatted prompts for generation and raw prompts for logging
        formatted_prompts = list(dataset["prompt"])
        raw_prompts = list(dataset["raw_prompt"]) if "raw_prompt" in dataset.column_names else formatted_prompts
        
        (
            prompt_input_ids, 
            generated_input_ids, 
            scores_view, 
            logits_view
        ) = generate_model_outputs(
            prompts=formatted_prompts,
            model=model,
            tokenizer=tokenizer,
            gen_kwargs=self.gen_kwargs,
        )

        for formatted_prompt, raw_prompt, pr_ids, gen_ids, scores, logits in zip(
            formatted_prompts,
            raw_prompts,
            prompt_input_ids, generated_input_ids,
            scores_view, logits_view, strict=True
        ):
            # Use formatted prompt as cache key (maintains consistency with training)
            cache_slot = self.cache[step].setdefault(formatted_prompt, [])
            pr_id = pr_ids[0]
            results = []
            
            # For each generated sequence for the same prompt
            for gen_id, score, logit in zip(
                gen_ids, scores, logits, strict=True
            ):
                generation_text = tokenizer.decode(gen_id, skip_special_tokens=True)
                
                formatted_response = format_response(
                    prompt=raw_prompt,
                    response=generation_text, 
                    tokenizer=tokenizer
                )
                
                cache_slot.append(
                    {
                        "score": self.estimator(gen_id, score, logit),
                        "prompt_input_ids": pr_id,
                        "generated_input_ids": gen_id,
                        "formatted_response": formatted_response,  # Can be used as chosen or rejected
                    }
                )
                
                results.append(generation_text)

            logging["generations"][raw_prompt] = results
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.tracker is not None:
            self.tracker.add_generations(logging)

    def _collect_sample_metadata(self, examples: list[dict[str, Any]]) -> tuple:
        """
        Count available pairs and collect metadata for lazy generation.
        
        Returns:
            Tuple of (expert_meta, replay_meta, noisy_meta, expert_count, replay_count, noisy_count)
        """
        expert_meta = []  # (example_idx, rejected_idx)
        replay_meta = []  # (example_idx, step, rejected_idx)
        noisy_meta  = []  # (prompt, step_a, step_b, curr_idx, past_idx, curr_score, past_score)
        
        for ex_idx, example in enumerate(examples):
            prompt_text = example["prompt"]
            
            current_step_data = self.cache.get(self.sampled_step, {}).get(prompt_text, [])
            
            # Collect expert samples (current step)
            for rej_idx in range(len(current_step_data)):
                expert_meta.append((ex_idx, rej_idx))
            
            # Collect replay samples (previous steps)
            for step_a in self.cache.keys():
                if step_a < self.sampled_step:
                    past_data = self.cache[step_a].get(prompt_text, [])
                    for rej_idx in range(len(past_data)):
                        replay_meta.append((ex_idx, step_a, rej_idx))
            
            # Collect noisy samples
            for step_a in self.cache.keys():
                step_data = self.cache.get(step_a, {})
                if prompt_text not in step_data:
                    continue
                
                current_entries = self.cache[step_a][prompt_text]
                
                for step_b in range(step_a):
                    past_entries = self.cache.get(step_b, {}).get(prompt_text)
                    if not past_entries:
                        continue
                    
                    for curr_idx, current in enumerate(current_entries):
                        for past_idx, past in enumerate(past_entries):
                            curr_score = float(current["score"])
                            past_score = float(past["score"])
                            
                            # Rejection sampling
                            margin = abs(curr_score - past_score)
                            if margin < self.rejection_thresh:
                                continue
                            
                            noisy_meta.append((
                                prompt_text, step_a, step_b, 
                                curr_idx, past_idx, 
                                curr_score, past_score
                            ))
        
        return (
            expert_meta, replay_meta, noisy_meta,
            len(expert_meta), len(replay_meta), len(noisy_meta)
        )

    def _generate_samples_from_indices(
        self, 
        examples: list[dict[str, Any]],
        expert_meta: list,
        replay_meta: list,
        noisy_meta: list,
        expert_indices: list[int],
        replay_indices: list[int],
        noisy_indices: list[int],
    ) -> tuple[list[tuple], list[tuple]]:
        """
        Generate actual sample pairs from sampled indices.
        
        Returns:
            Tuple of (samples, noisy_samples_for_tracking)
        """
        samples = []
        
        # Generate expert samples
        for idx in expert_indices:
            ex_idx, rej_idx = expert_meta[idx]
            example = examples[ex_idx]
            current_step_data = self.cache[self.sampled_step][example["prompt"]]
            rejected = current_step_data[rej_idx]
            
            samples.append((
                example["prompt_input_ids"],
                example["chosen_input_ids"],
                rejected["generated_input_ids"],
            ))
        
        # Generate replay samples
        for idx in replay_indices:
            ex_idx, step_a, rej_idx = replay_meta[idx]
            example = examples[ex_idx]
            past_data = self.cache[step_a][example["prompt"]]
            rejected = past_data[rej_idx]
            
            samples.append((
                example["prompt_input_ids"],
                example["chosen_input_ids"],
                rejected["generated_input_ids"],
            ))
        
        # Generate noisy samples and track for logging
        noisy_samples_for_tracking = []
        
        for idx in noisy_indices:
            prompt_text, step_a, step_b, curr_idx, past_idx, curr_score, past_score = noisy_meta[idx]
            
            current = self.cache[step_a][prompt_text][curr_idx]
            past = self.cache[step_b][prompt_text][past_idx]
            prompt_input_ids = current["prompt_input_ids"]
            
            past_is_better = (
                (past_score > curr_score) 
                if self.higher_is_better 
                else (past_score < curr_score)
            )
            chosen = past if past_is_better else current
            rejected = current if past_is_better else past
            
            samples.append((
                prompt_input_ids,
                chosen["generated_input_ids"],
                rejected["generated_input_ids"],
            ))
            
            noisy_samples_for_tracking.append((
                chosen["generated_input_ids"],
                rejected["generated_input_ids"],
                float(chosen["score"]),
                float(rejected["score"])
            ))
        
        return samples, noisy_samples_for_tracking

    def _build_batch(self, samples: list[tuple]) -> dict[str, torch.Tensor]:
        """
        Build the final batch dictionary from samples.
        
        Args:
            samples: List of (prompt_ids, chosen_ids, rejected_ids) tuples
        
        Returns:
            Batch dictionary with padded tensors and attention masks
        """
        def attn_mask(input_ids: list[torch.Tensor]) -> list[torch.Tensor]:
            return [torch.ones_like(input_id) for input_id in input_ids]
        
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

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Main collator function that creates preference pairs for DPO training.
        Uses lazy evaluation to avoid OOM on large cache sizes.
        """
        (
            expert_meta, replay_meta, noisy_meta,
            expert_count, replay_count, noisy_count
        ) = self._collect_sample_metadata(examples)
        
        # Sample indices
        n_expert = int(expert_count * self.frac_expert)
        n_replay = int(replay_count * self.frac_replay)
        n_noisy = int(noisy_count * self.frac_noisy)
        
        expert_indices = random.sample(range(expert_count), min(expert_count, n_expert)) if expert_count > 0 else []
        replay_indices = random.sample(range(replay_count), min(replay_count, n_replay)) if replay_count > 0 else []
        noisy_indices = random.sample(range(noisy_count), min(noisy_count, n_noisy)) if noisy_count > 0 else []
        
        # Lazy generate sample pairs
        samples, noisy_samples_for_tracking = self._generate_samples_from_indices(
            examples, expert_meta, replay_meta, noisy_meta,
            expert_indices, replay_indices, noisy_indices
        )
        
        # Track noisy samples
        if self.tracker is not None and noisy_samples_for_tracking:
            tracker_logging = {
                "sampled_data": [
                    (self.tokenizer.decode(chosen), self.tokenizer.decode(rejected))
                    for chosen, rejected, _, _
                    in noisy_samples_for_tracking
                ],
                "uncertainty": [
                    (chosen_score, rejected_score) 
                    for _, _, chosen_score, rejected_score 
                    in noisy_samples_for_tracking
                ],
                "margin": [
                    abs(chosen_score - rejected_score)
                    for _, _, chosen_score, rejected_score 
                    in noisy_samples_for_tracking
                ],
            }
            self.tracker.add_collator_sampling(tracker_logging)
        
        if not samples and expert_count > 0:
            ex_idx, rej_idx = expert_meta[0]
            example = examples[ex_idx]
            current_step_data = self.cache[self.sampled_step][example["prompt"]]
            rejected = current_step_data[rej_idx]
            
            samples = [(
                example["prompt_input_ids"],
                example["chosen_input_ids"],
                rejected["generated_input_ids"],
            )]
        
        return self._build_batch(samples)
