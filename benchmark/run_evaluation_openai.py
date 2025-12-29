import os
import csv
import json
import logging
import textwrap
from typing import Any
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from itertools import permutations
from pydantic import BaseModel, Field

import hydra
import openai
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = ROOT / "outputs" / "generations"
EXAMPLES_DIR = GEN_DIR / "examples"
OUTPUT_PATH = ROOT / "outputs" / "evaluations"

class JudgeResult(BaseModel):
    answer: Literal["A", "B"] = Field(description="The option most similar to the HUMAN AUTHOR'S WRITING; either A or B")
    reasoning: str = Field(description="Brief explanation of why this option is more similar")

PROMPT_TEMPLATE = textwrap.dedent(
    """You are an impartial evaluator.
    Below is a sample of a human author"s writing and two options.

    ### HUMAN AUTHOR"S WRITING:
    {demo}

    ### OUTPUT A:
    {text_a}

    ### OUTPUT B:
    {text_b}

    ### Task
    Which option was written by the human author based on similarity to the HUMAN AUTHOR"S WRITING above?

    ALWAYS REMAIN IMPARTIAL WHEN EVALUATING OUTPUTS.
    """
)

def aggregate_responses() -> dict[str, Any]:
    data = {}

    def read_prompts(csv_path: Path):
        prompts, completions = [], []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            
            headers = reader.fieldnames or []
            if "completion" in headers:
                resp_key = "completion"
            elif "demo" in headers:
                resp_key = "demo"
            elif "chosen" in headers:
                resp_key = "chosen"
            else:
                resp_key = headers[1] if len(headers) > 1 else "completion"

            for row in reader:
                p_text = row.get("prompt", "").strip()
                c_text = row.get(resp_key, "").strip()

                if not p_text and not c_text:
                    continue
                
                prompts.append(p_text)
                completions.append(c_text)
                
        return prompts, completions

    for folder in sorted(GEN_DIR.iterdir()):
        if not folder.is_dir() or folder.name == "examples":
            continue
        
        parts = folder.name.split("_")
        group_id, model_info = f"{parts[0]}_{parts[1]}", "_".join(parts[2:])

        if group_id not in data:
            prompts, demos = read_prompts(EXAMPLES_DIR / f"{group_id}.csv")
            data[group_id] = {"prompt": prompts, "demo": demos, "generations": {}}

        entry = data[group_id]
        expected_len = len(entry["prompt"])

        for csv_file in sorted(folder.glob("*.csv")):
            gen_prompts, gen_completions = read_prompts(csv_file)
            actual_len = len(gen_prompts)

            if actual_len != expected_len:
                if actual_len > expected_len:
                    logger.warning(
                        f"MISMATCH FIXED: {csv_file.name} has {actual_len} prompts, "
                        f"truncating to expected {expected_len}."
                    )
                    gen_completions = gen_completions[:expected_len]
                else:
                    logger.error(
                        f"MISMATCH: {csv_file.name} has {actual_len} prompts, "
                        f"but expected {expected_len}. Skipping."
                    )
                    continue
            
            entry["generations"][f"{model_info}_{csv_file.stem}"] = gen_completions

    comparison_output = OUTPUT_PATH / "comparison.json"
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    with comparison_output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)

    logger.info("Finished aggregating data.")
    return data

def create_jsonl(data: dict[str, Any], config) -> None:
    requests = []

    for data_aid, content in data.items():
        prompts_list = content["prompt"]
        demos_list = content["demo"]
        generations_dict = content["generations"]
        
        model_names = list(generations_dict.keys())

        for model_name_a, model_name_b in permutations(model_names, 2):
            
            completions_a = generations_dict[model_name_a]
            completions_b = generations_dict[model_name_b]

            for p_text, d_text, resp_a, resp_b in zip(prompts_list, demos_list, completions_a, completions_b, strict=True):
                
                context_input = f"{p_text}\n{d_text}"
                option_a = f"{p_text}\n{resp_a}"
                option_b = f"{p_text}\n{resp_b}"

                requests.append({
                    "custom_id": f"{data_aid}__{model_name_a}_VS_{model_name_b}", # ccat50_0__Ministral-8B-Instruct-2410_msp_ditto_VS_Ministral-8B-Instruct-2410_None_ditto
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": config.model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": PROMPT_TEMPLATE.format(demo=context_input, text_a=option_a, text_b=option_b)},
                        ],
                    },
                })

    comparison_output = OUTPUT_PATH / "batch-request.jsonl"
    comparison_output.parent.mkdir(parents=True, exist_ok=True)

    with comparison_output.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info("Finished created jsonl file.")

def create_batch_request(config) -> None:
    client = openai.Client()

    uploaded_file = client.files.upload(
        file=OUTPUT_PATH / "batch-request.jsonl",
        purpose="batch",
    )

    file_batch_job = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    logger.info(f"=>> Created batch job: {file_batch_job.name}")



@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    load_dotenv()
    os.environ["api_key"] = os.environ.get("OPENAI_API_KEY")
    logger.info("=>> Starting the evaluation process...")
    logger.info(f"Model for evaluation: {config.model_name}")
    data = aggregate_responses()
    create_jsonl(data, config)
    # create_batch_request(config)


if __name__ == "__main__":
    main()
