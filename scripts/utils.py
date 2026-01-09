import random
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict


def format_response(prompt, response, tokenizer, *, add_bos=True):
    prompt_msgs = [{"role": "user", "content": prompt}]
    response_msgs = [{"role": "assistant", "content": response}]
    full_conversation = prompt_msgs + response_msgs
    
    formatted_text = tokenizer.apply_chat_template(full_conversation, tokenize=False, add_generation_prompt=False)
    
    return formatted_text


def format_for_training(prompt, chosen, tokenizer, mode="sft"):
    if mode == "sft":
        # SFT: return full conversation as-is from chat template
        return {"text": format_response(prompt, chosen, tokenizer, add_bos=False)}
    
    # DPO: separate prompt and chosen, let tokenizer handle special tokens
    prompt_msgs = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    
    return {
        "prompt": prompt_text,
        "chosen": format_response(prompt, chosen, tokenizer, add_bos=False),
        "rejected": "",  # Will be filled during training
    }


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def clone_adapter(model: PeftModel, src_name: str, tgt_name: str) -> None:
    model.add_adapter(tgt_name, model.peft_config[src_name])
    src_weights = get_peft_model_state_dict(model, adapter_name=src_name)
    tgt_weights = {k.replace(f".{src_name}.", f".{tgt_name}."): v.clone() for k, v in src_weights.items()}
    set_peft_model_state_dict(model, tgt_weights, adapter_name=tgt_name)

def generate_model_outputs(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    gen_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:    
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    prompt_ids_list, gen_ids_list, scores_list, logits_list = [], [], [], []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                **gen_kwargs,
            )

        sequences = outputs.sequences.detach().cpu()              # [n_ret, total_len]
        scores = [s.detach().cpu() for s in outputs.scores]       # list of [n_ret, vocab]
        del outputs

        prompt_ids = sequences[:, :prompt_len].contiguous()       # [n_ret, prompt_len]
        gen_ids = sequences[:, prompt_len:].contiguous()          # [n_ret, gen_len]

        logits = torch.stack(scores, dim=1)                       # [n_ret, gen_len, vocab]
        trans_scores = model.compute_transition_scores(sequences, scores, normalize_logits=False)

        prompt_ids_list.append(prompt_ids)
        gen_ids_list.append(gen_ids)
        scores_list.append(trans_scores)
        logits_list.append(logits)
    
    # Pad to batch
    pad_id = tokenizer.pad_token_id or 0
    max_prompt_len = max(t.size(1) for t in prompt_ids_list)
    max_gen_len = max(t.size(1) for t in gen_ids_list)
    
    prompt_input_ids = torch.stack([F.pad(t, (0, max_prompt_len - t.size(1)), value=pad_id) for t in prompt_ids_list], dim=0)
    generated_input_ids = torch.stack([F.pad(t, (0, max_gen_len - t.size(1)), value=pad_id) for t in gen_ids_list], dim=0)

    max_score_len = max(t.size(1) for t in scores_list)
    scores_view = torch.stack([F.pad(t, (0, max_score_len - t.size(1)), value=0.0) for t in scores_list], dim=0)
    logits_view = torch.stack([F.pad(t, (0, 0, 0, max_score_len - t.size(1)), value=0.0) for t in logits_list], dim=0)

    return prompt_input_ids, generated_input_ids, scores_view, logits_view
