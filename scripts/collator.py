# DITTO Authors: Omar Shaikh, Michelle S. Lam, Joey Hejna, Yijia Shao, Hyundong Cho, Michael S. Bernstein, Diyi Yang
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.

import gc
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

def check_gpu_memory(label=""):
    if not torch.cuda.is_available():
        return
    device = torch.device('cuda:0')
    
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    print(f"[{label}] Memory Used: {mem_used_MB:.2f} MB")
    return mem_used_MB

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
    cache: dict[int, dict[str, list[dict[str, torch.Tensor]]]] = field(default_factory=dict)
    sampled_step: int = field(default=0, init=False)

    def __post_init__(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to `DITTOCollator`.")
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.pad_token_id

    def _get_cache_entries(self, step: int, prompt: str) -> list[dict]:
        """Helper to safely get cache entries for a given step and prompt."""
        return self.cache.get(step, {}).get(prompt, [])

    def resample(
        self,
        step: int,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset | IterableDataset,
    ) -> None:
        check_gpu_memory("Resample: Start")

        self.sampled_step = step
        self.cache.setdefault(step, {})

        logging_dict = {"generations": {}}
        formatted_prompts = list(dataset["prompt"])
        raw_prompts = list(dataset["raw_prompt"]) if "raw_prompt" in dataset.column_names else formatted_prompts
        
        generation_results = generate_model_outputs(
            prompts=formatted_prompts,
            model=model,
            tokenizer=tokenizer,
            gen_kwargs=self.gen_kwargs,
        )

        check_gpu_memory("Resample: After Gen (Peak)")

        for i, (formatted_prompt, raw_prompt) in enumerate(zip(formatted_prompts, raw_prompts)):
            
            pr_ids_batch, gen_ids_batch, scores_batch, logits_batch = generation_results[i]
            
            cache_slot = self.cache[step].setdefault(formatted_prompt, [])
            tracker_results_for_prompt = []

            for j in range(len(gen_ids_batch)):
                
                single_gen_id = gen_ids_batch[j]
                single_score = scores_batch[j]
                single_logit = logits_batch[j]

                # --- Compute Estimator ---
                estimator_score = self.estimator(
                    input_ids=single_gen_id, 
                    logprobs=single_score, 
                    logits=single_logit
                )
                if isinstance(estimator_score, torch.Tensor):
                    estimator_score = estimator_score.item()

                raw_gen_list = single_gen_id.tolist()
                raw_pr_list = pr_ids_batch[j].tolist()
                
                saved_gen_ids = [t for t in raw_gen_list if t != self.pad_token_id]
                saved_pr_ids = [t for t in raw_pr_list if t != self.pad_token_id]

                generation_text = tokenizer.decode(saved_gen_ids, skip_special_tokens=True)
                
                formatted_response = format_response(
                    prompt=raw_prompt,
                    response=generation_text, 
                    tokenizer=tokenizer
                )
                
                cache_slot.append(
                    {
                        "score": estimator_score,
                        "prompt_input_ids": saved_pr_ids,
                        "generated_input_ids": saved_gen_ids,
                        "formatted_response": formatted_response,
                    }
                )
                
                tracker_results_for_prompt.append(generation_text)

            logging_dict["generations"][raw_prompt] = tracker_results_for_prompt
        
        del generation_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.tracker is not None:
            self.tracker.add_generations(logging_dict)

        check_gpu_memory("Resample: End (Cleanup Done)")

    def _collect_sample_metadata(self, examples: list[dict[str, Any]]) -> tuple:
        """
        Count available pairs and collect metadata for lazy generation.
        
        Returns:
            Tuple of (expert_meta, replay_meta, noisy_meta, expert_count, replay_count, noisy_count)
        """
        expert_meta = []  # (example_idx, rejected_idx)
        replay_meta = []  # (example_idx, step, rejected_idx)
        noisy_meta = []   # (prompt, step_a, step_b, curr_idx, past_idx, curr_score, past_score)
        
        for ex_idx, example in enumerate(examples):
            prompt_text = example["prompt"]
            
            # Collect expert samples (current step)
            current_step_data = self._get_cache_entries(self.sampled_step, prompt_text)
            expert_meta.extend((ex_idx, rej_idx) for rej_idx in range(len(current_step_data)))
            
            # Collect replay samples (previous steps)
            for step in self.cache:
                if step < self.sampled_step:
                    past_data = self._get_cache_entries(step, prompt_text)
                    replay_meta.extend((ex_idx, step, rej_idx) for rej_idx in range(len(past_data)))
            
            # Collect noisy samples
            for step_a, step_data in self.cache.items():
                if not (current_entries := step_data.get(prompt_text)):
                    continue
                
                for step_b in range(step_a):
                    if not (past_entries := self._get_cache_entries(step_b, prompt_text)):
                        continue
                    
                    for curr_idx, current in enumerate(current_entries):
                        curr_score = float(current["score"])
                        for past_idx, past in enumerate(past_entries):
                            past_score = float(past["score"])
                            if abs(curr_score - past_score) >= self.rejection_thresh:
                                noisy_meta.append((
                                    prompt_text, step_a, step_b,
                                    curr_idx, past_idx,
                                    curr_score, past_score
                                ))
        
        return (expert_meta, replay_meta, noisy_meta, len(expert_meta), len(replay_meta), len(noisy_meta))

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
        samples = []
        sample_types = []
        noisy_samples_for_tracking = []

        # Generate expert samples
        for idx in expert_indices:
            ex_idx, rej_idx = expert_meta[idx]
            example = examples[ex_idx]
            rejected_cache = self.cache[self.sampled_step][example["prompt"]][rej_idx]
            
            prompt_ids = list(example["prompt_input_ids"])
            chosen_ids = list(example["chosen_input_ids"])
            # Use generated_input_ids directly (completion only) and prepend example's prompt
            rejected_gen_ids = list(rejected_cache["generated_input_ids"])
            
            samples.append((
                prompt_ids,
                prompt_ids + chosen_ids,  # full chosen
                prompt_ids + rejected_gen_ids,  # full rejected (using example's prompt)
            ))
            sample_types.append(0)

        # Generate replay samples
        for idx in replay_indices:
            ex_idx, step_a, rej_idx = replay_meta[idx]
            example = examples[ex_idx]
            rejected_cache = self.cache[step_a][example["prompt"]][rej_idx]
            
            prompt_ids = list(example["prompt_input_ids"])
            chosen_ids = list(example["chosen_input_ids"])
            rejected_gen_ids = list(rejected_cache["generated_input_ids"])
            
            samples.append((
                prompt_ids,
                prompt_ids + chosen_ids,
                prompt_ids + rejected_gen_ids,
            ))
            sample_types.append(1)

        # Generate noisy samples
        for idx in noisy_indices:
            prompt_text, step_a, step_b, curr_idx, past_idx, curr_score, past_score = noisy_meta[idx]
            current = self.cache[step_a][prompt_text][curr_idx]
            past = self.cache[step_b][prompt_text][past_idx]

            past_is_better = (past_score > curr_score) if self.higher_is_better else (past_score < curr_score)
            chosen_cache, rejected_cache = (past, current) if past_is_better else (current, past)
            
            matching_example = next((ex for ex in examples if ex["prompt"] == prompt_text), None)
            if matching_example is not None:
                prompt_ids = list(matching_example["prompt_input_ids"])
            else:
                prompt_ids = list(current["prompt_input_ids"])
            
            chosen_gen_ids = list(chosen_cache["generated_input_ids"])
            rejected_gen_ids = list(rejected_cache["generated_input_ids"])

            samples.append((
                prompt_ids,
                prompt_ids + chosen_gen_ids,
                prompt_ids + rejected_gen_ids,
            ))
            sample_types.append(2)
            noisy_samples_for_tracking.append((
                chosen_cache["generated_input_ids"],
                rejected_cache["generated_input_ids"],
                float(chosen_cache["score"]),
                float(rejected_cache["score"]),
            ))

        return samples, noisy_samples_for_tracking, sample_types

    def _build_batch(self, samples: list[tuple], sample_types: list[int]) -> dict[str, torch.Tensor]:
        prompt_ids_list = []
        chosen_completion_ids_list = []
        rejected_completion_ids_list = []

        for p_ids, c_full_ids, r_full_ids in samples:
            p_ids = list(p_ids) if not isinstance(p_ids, list) else p_ids
            c_full_ids = list(c_full_ids) if not isinstance(c_full_ids, list) else c_full_ids
            r_full_ids = list(r_full_ids) if not isinstance(r_full_ids, list) else r_full_ids
            
            prompt_len = len(p_ids)
            
            if len(c_full_ids) > prompt_len:
                chosen_completion = c_full_ids[prompt_len:]
            else:
                chosen_completion = c_full_ids
                
            if len(r_full_ids) > prompt_len:
                rejected_completion = r_full_ids[prompt_len:]
            else:
                rejected_completion = r_full_ids
            
            if len(chosen_completion) == 0:
                chosen_completion = [self.tokenizer.eos_token_id]
            if len(rejected_completion) == 0:
                rejected_completion = [self.tokenizer.eos_token_id]
            
            prompt_ids_list.append(torch.tensor(p_ids, dtype=torch.long))
            chosen_completion_ids_list.append(torch.tensor(chosen_completion, dtype=torch.long))
            rejected_completion_ids_list.append(torch.tensor(rejected_completion, dtype=torch.long))

        batch = {}
        
        batch["prompt_input_ids"] = pad(
            prompt_ids_list, 
            padding_value=self.tokenizer.pad_token_id, 
            padding_side="left"
        )
        batch["prompt_attention_mask"] = pad(
            [torch.ones_like(t) for t in prompt_ids_list],
            padding_value=0,
            padding_side="left",
        )
        
        batch["chosen_input_ids"] = pad(
            chosen_completion_ids_list, 
            padding_value=self.tokenizer.pad_token_id, 
            padding_side="right"
        )
        batch["chosen_attention_mask"] = pad(
            [torch.ones_like(t) for t in chosen_completion_ids_list],
            padding_value=0,
            padding_side="right",
        )
        
        batch["rejected_input_ids"] = pad(
            rejected_completion_ids_list, 
            padding_value=self.tokenizer.pad_token_id, 
            padding_side="right"
        )
        batch["rejected_attention_mask"] = pad(
            [torch.ones_like(t) for t in rejected_completion_ids_list],
            padding_value=0,
            padding_side="right",
        )
        batch["sample_types"] = torch.tensor(sample_types, dtype=torch.long)

        return batch

    def _sample_indices(self, count: int, n_samples: int) -> list[int]:
        """Sample indices from range, handling empty case."""
        return random.sample(range(count), min(count, n_samples)) if count > 0 else []

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Main collator function that creates preference pairs for DPO training.
        Uses lazy evaluation to avoid OOM on large cache sizes.
        """
        expert_meta, replay_meta, noisy_meta, expert_count, replay_count, noisy_count = \
            self._collect_sample_metadata(examples)

        len_superbatch = self.rescale_batch * len(examples)
        n_expert = round(len_superbatch * self.frac_expert)
        n_replay = round(len_superbatch * self.frac_replay)
        n_noisy = len_superbatch - n_expert - n_replay

        samples, noisy_samples_for_tracking, sample_types = self._generate_samples_from_indices(
            examples, expert_meta, replay_meta, noisy_meta,
            self._sample_indices(expert_count, n_expert),
            self._sample_indices(replay_count, n_replay),
            self._sample_indices(noisy_count, n_noisy),
        )

        # Track noisy samples
        if self.tracker and noisy_samples_for_tracking:
            self.tracker.add_collator_sampling({
                "sampled_data": [
                    (self.tokenizer.decode(c), self.tokenizer.decode(r))
                    for c, r, _, _ in noisy_samples_for_tracking
                ],
                "uncertainty": [
                    (cs, rs) for _, _, cs, rs in noisy_samples_for_tracking
                ],
                "margin": [
                    abs(cs - rs) for _, _, cs, rs in noisy_samples_for_tracking
                ],
            })

        return self._build_batch(samples, sample_types)
