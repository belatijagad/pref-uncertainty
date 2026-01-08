import gc
import os
import logging
from pathlib import Path
from typing import cast

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import hydra
from omegaconf import OmegaConf
import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOConfig, SFTConfig, SFTTrainer
from huggingface_hub import repo_exists, file_exists

from scripts.trainer import DITTOTrainer
from scripts.callback import (
    EarlyStoppingCallback,
    ResampleCallback,
)
from scripts.collator import DITTOCollator
from scripts.utils import (
    clone_adapter,
    format_for_training,
    seed_everything,
)
from scripts.estimator import ESTIMATOR_MAP
from scripts.tracker import Tracker



logger = logging.getLogger(__name__)
logging.getLogger("transformers.pipelines").setLevel(logging.WARNING)


def load_author_subset(config):
    """Load and trim the dataset to the configured author/sample count."""
    raw_dataset = (
        load_dataset(config.dataset["name_or_path"])["train"]
        .filter(lambda x: x["author_id"] == config.dataset.author_id)
        .shuffle(seed=config.seed)
    )

    num_samples = min(config.dataset.train_samples_per_author, len(raw_dataset))
    return raw_dataset.select(range(num_samples))


def build_sft_dataset(raw_dataset, tokenizer):
    """Build SFT dataset by formatting prompt + chosen with chat template."""
    return raw_dataset.map(
        lambda x: format_for_training(x["prompt"], x["chosen"], tokenizer, mode="sft"),
        remove_columns=raw_dataset.column_names,
    )


def build_dpo_dataset(raw_dataset, tokenizer):
    """Build DPO dataset by formatting prompt and chosen separately."""
    def format_with_raw(example):
        formatted = format_for_training(example["prompt"], example["chosen"], tokenizer, mode="dpo")
        formatted["raw_prompt"] = example["prompt"]
        return formatted
    
    return raw_dataset.map(format_with_raw)
        
@hydra.main(version_base=None, config_path="../configs", config_name="ditto")
def main(config: DictConfig):
    load_dotenv()
    seed_everything(config["seed"])

    run_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']

    # Prepare model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model["name_or_path"],
        attn_implementation=config.model["attn_implementation"],
        dtype=torch.bfloat16 if config.model["use_bf16"] else torch.float16,
        device_map="auto",
    )
    lora_config = LoraConfig(**OmegaConf.to_container(config.lora, resolve=True), task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config, adapter_name="ref_model")
    model.set_adapter("ref_model")
    tokenizer = AutoTokenizer.from_pretrained(config.model["name_or_path"])
    
    # Add a dedicated padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Added new pad token '<pad>' (id: {tokenizer.pad_token_id})")
    
    # Ensure model config is updated
    model.config.pad_token_id = tokenizer.pad_token_id

    # 1. Load Raw Data
    raw_dataset = load_author_subset(config)

    # 2. Prepare SFT Dataset (Uses format_sft -> Returns "text")
    logger.info("Formatting dataset for SFT...")
    sft_dataset = build_sft_dataset(raw_dataset, tokenizer)

    # 3. Prepare DPO Dataset (Uses format_dpo_smart -> Returns prompt/chosen/rejected)
    logger.info("Formatting dataset for DITTO/DPO...")
    dpo_dataset = build_dpo_dataset(raw_dataset, tokenizer)

    if enable_wandb := config.wandb["enabled"]:
        config.wandb.__delattr__("enabled")
        wandb.init(**config.wandb)

    # Train SFT
    tokenizer.padding_side = "right"
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_dataset,
        args=SFTConfig(
            output_dir=run_dir,
            report_to="wandb" if enable_wandb else "none",
            chat_template_path=config.model["name_or_path"],
            dataset_text_field="text",
            **config.training_args.sft,
            **config.training_args.general,
        ),
        optimizer_cls_and_kwargs=(AdamW, config.optim_args.sft),
        callbacks=[EarlyStoppingCallback(threshold=1.0)],
    )
    trainer.train()
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # Test all uncertainty methods at once
    repo_username = "belati"
    repo_name_base = f"{config.model.name}_{config.dataset.name}_{config.dataset.author_id}"
    full_repo_id = f"{repo_username}/{repo_name_base}"

    tokenizer.padding_side = "left"
    for name, estimator in ESTIMATOR_MAP.items():
        logger.info(f"Preparing DITTO training for method: {name}")
        
        adapter_name = f"{name}_policy_model"

        if repo_exists(full_repo_id):
            if file_exists(full_repo_id, filename="adapter_config.json"):
                logger.info(f"=>> Method {name} already exists in {full_repo_id}; skipping...")
                continue
        
        model.set_adapter("ref_model")
        clone_adapter(cast(PeftModel, model), "ref_model", adapter_name)
        
        model.set_adapter(adapter_name)
        
        # Log generations, uncertainty score
        tracker = Tracker(run_dir=Path("logs") / f"{config.dataset.name}_{config.dataset.author_id}" / f"{name}")
        data_collator = DITTOCollator(
            **config.sampler,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            estimator=estimator,
            higher_is_better=estimator.higher_is_better,
            tracker=tracker,
        )
        
        dpo_trainer = DITTOTrainer(
            model=model,
            args=DPOConfig(
                output_dir=str(Path(run_dir) / name),
                report_to="wandb" if enable_wandb else "none",
                model_adapter_name=adapter_name,
                ref_adapter_name="ref_model",
                push_to_hub=config.push_to_hub,
                hub_model_id=full_repo_id,
                remove_unused_columns=False,
                **config.training_args.dpo,
                **config.training_args.general,
            ),
            optimizer_cls_and_kwargs=(AdamW, config.optim_args.dpo),
            processing_class=tokenizer,
            train_dataset=dpo_dataset,
            data_collator=data_collator,
            callbacks=[
                ResampleCallback(
                    model, tokenizer, dpo_dataset, data_collator, config.sampler
                ),
            ],
            # Track results
            tracker=tracker,
        )
        
        dpo_trainer.train()
        dpo_trainer.save_model()
        tracker.save()
        
        # Cleanup
        del dpo_trainer
        model.delete_adapter(adapter_name)
        gc.collect()
        torch.cuda.empty_cache()
        
        # Switch back to ref for next cloning
        model.set_adapter("ref_model")

if __name__ == "__main__":
    main()
