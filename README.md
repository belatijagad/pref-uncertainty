# Preference Uncertainty

---

## Installation

Install `uv` package manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment using `uv` and install the project dependencies.

```bash
uv venv -p 3.12
source .venv/bin/activate
uv sync
```

Create `.env` to store token credentials for `huggingface_hub` and `wandb`

```bash
HUGGINGFACE_API_KEY=
WANDB_API_KEY=
GEMINI_API_KEY=
```

## Usage

The output directory will be structured this way.

```bash
outputs/
├── models/
│   └── {dataset}_{author_id}_{model name}/
│       ├── ref_model
│       └── {method_name}_model
├── generations/
│   ├── {dataset}_{author_id}_{model_name}/
│   │   ├── zero_shot.csv
│   │   ├── few_shots.csv
│   │   ├── sft.csv
│   │   └── dpo.csv
│   └── examples/
│       └── {dataset}_{author_id}_{model_name}.csv
└── evals/
    ├── comparisons.json
    ├── batch-request.jsonl
    ├── batch-results.json
    ├── match_statistics.json
    └── winrate.json
```

### Training

```bash
uv run scripts/train_ditto.py \
    wandb.enabled=false model.name_or_path={model of choice}
```

It will save the model.

```bash
models/
└── {dataset}_{author_id}_{model name}/
    ├── ref_model
    └── {method_name}_model
```


### Evaluation

Generate the responses for evaluation

```bash
uv run generate_samples.py
```

```bash
generations/
├── {dataset}_{author_id}_{model_name}/
│   ├── zero_shot.csv
│   ├── few_shots.csv
│   ├── sft.csv
│   └── ditto.csv
└── examples/
    └── {dataset}_{author_id}_{model_name}.csv
```

Score the results using LLM judge

```bash
uv run evaluate.py
```


```bash
evals/
├── comparisons.json
├── batch-request.jsonl
├── batch-results.json
├── match_statistics.json
└── winrate.json
```