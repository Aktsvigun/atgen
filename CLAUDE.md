# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

ATGen (Active Learning for Text Generation) is a toolkit for applying active learning strategies to NLP/text-generation tasks. The core idea: intelligently select which unlabeled examples to annotate each round, minimizing annotation cost while maximizing model performance. It supports multiple selection strategies, labellers (human, LLM API, golden), and evaluation metrics.

## Commands

```bash
# Install for development
pip install -e .

# Run active learning experiment (default model: Qwen/Qwen3-1.7B, dataset: aeslc, strategy: random)
run-al al.strategy=huds data=xsum model.checkpoint=Qwen/Qwen2.5-7B al.query_size=10

# Run subset selection (one-shot: select k examples, train once, evaluate)
run-ss

# Select a specific config file via env var (e.g. test.yaml)
HYDRA_CONFIG_NAME=test run-al +save_model=false

# Suppress pdb on exception (debug=true by default)
run-al +debug=false

# Launch Streamlit dashboard
streamlit run Welcome.py

# Run tests
pytest

# Run a single test
pytest test/test_benchmark.py::test_just_works
```

## Linting / Formatting

Pre-commit hooks are configured (`.pre-commit-config.yaml`):
- **Ruff** (lint + format, auto-fix)
- **MyPy** (strict, `--ignore-missing-imports`)

Run manually: `pre-commit run --all-files`

## Architecture

### Core Loop (`src/atgen/run_scripts/run_active_learning.py`)

Each AL iteration: **train model → generate on unlabeled pool → query strategy selects top-k → labeller annotates → evaluate**. The loop repeats for `al.num_iterations` rounds. iter_0 is an optional pre-training evaluation pass; iterations 1..N do the full cycle.

`run_subset_selection.py` is a simpler variant: select k examples once, train once, evaluate — no iterative loop. It also accepts `al.query_ids_path` to skip strategy selection and load pre-computed IDs from a JSON file.

### Key Modules

| Module | Role |
|---|---|
| `strategies/` | AL query selection algorithms. All implement `__call__(unlabeled_pool, num_to_label, ...) -> list[int]`. Factory: `get_strategy.py` |
| `labellers/` | Annotation sources: `golden` (dataset GT), `human` (interactive UI), `custom_llm` (local model), `api_llm` (OpenAI/Anthropic). Factory: `get_labeller.py` |
| `metrics/` | Evaluation: ROUGE, BLEU, BERTScore, BARTScore, AlignScore, DeepEval LLM-based |
| `utils/` | Data loading, model/tokenizer loading (Unsloth), training setup, generation, embeddings |
| `run_scripts/` | Entry points: `run_active_learning.py` and `run_subset_selection.py` |

### Configuration System (Hydra)

All config lives in `configs/`. The main entry is `configs/base.yaml` with defaults `labeller=golden`, `data=aeslc`, `al=random`. Override any field on the CLI:

```bash
run-al al.strategy=hadas al.query_size=50 model.quantize=true
```

Select a config file with `HYDRA_CONFIG_NAME=<name>` (e.g., `test`). Override the config directory with `HYDRA_CONFIG_PATH`.

Config groups:
- `configs/al/` — strategy-specific params (each file sets `strategy:` and `strategy_kwargs:`)
- `configs/data/` — dataset definitions (xsum, aeslc, gsm8k, user_data, etc.)
- `configs/labeller/` — golden, human, api_llm, custom_llm

Key config sections: `al`, `model`, `training`, `inference`, `evaluation`, `data`, `labeller`.

`strategy_kwargs` in an `al/` config is passed directly to the strategy's `__init__`. For example `random_init: true` in `al/hadas.yaml` triggers warm-start random labeling before embedding-heavy selection.

Two custom Hydra resolvers are registered in `main_decorator.py` before config loads:
- `${multiply_with_few_shot:base_len,n}` — scales max token lengths when few-shot examples are added, avoiding OOM
- `${to_string:path}` — escapes model checkpoint paths for use as directory names

`validate_and_fill_config()` applies cross-field defaults after Hydra resolves the config (e.g., derives `train_output_column_name` / `test_output_column_name` from the data config, fills missing API keys by cascade).

### Entry Point Decoration

All entry points are wrapped with `@main_decorator` (`utils/main_decorator.py`), which:
1. Registers Hydra resolvers
2. Calls `validate_and_fill_config()`
3. Restores the original working directory after Hydra changes it
4. Disables wandb, sets seeds, optionally sets HuggingFace offline mode (`offline_mode=true`)
5. Saves resolved `config.yaml` to the output workdir

When `config.debug=true` (default), unhandled exceptions drop into `pdb`. Pass `+debug=false` on the CLI to suppress this.

### Training Stack

- **Unsloth** for fast LoRA fine-tuning (4-bit quantization via bitsandbytes)
- **PEFT** for LoRA adapters (default: r=32, lora_alpha=32)
- **TRL** SFTTrainer with optional packing and early stopping
- **vLLM** as default inference framework for unlabeled pool generation; also supports SgLang and plain transformers (selected via `inference.framework`)

### Active Learning Strategies

Registered strategies (in `strategies/__init__.py`): `hadas`, `huds`, `te_delfy`, `random`, `nsp`, `bleuvar`, `idds`, `dual`.

Add new strategies by subclassing `BaseStrategy` and registering in `strategies/__init__.py`.

`get_strategy.py` uses `inspect.getfullargspec()` to pass only kwargs the strategy's `__init__` accepts — strategies declare what they need rather than receiving the full config.

**Embedding-heavy strategies (HUDS, HADAS)** compute embeddings in `__init__`, not lazily. They are expensive to instantiate but fast to query. `BaseStrategy._select_subsample_if_necessary()` can subsample the unlabeled pool before embedding to bound memory.

`random_init` (set in `strategy_kwargs`) makes a strategy use random selection for the first AL iteration before switching to its own scoring logic.

### Output Column Name Duality

Train and test phases can use different output column names (`train_output_column_name` vs `test_output_column_name`), derived in `validate_and_fill_config()`. This lets you use different prompt formats or target fields per phase without touching strategy/labeller code.

### API Key Resolution

Evaluation API keys are resolved in this order: `config.evaluation.api_key` → `EVALUATION_API_KEY` env var → provider-specific env var (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) → `config.labeller.api_key`. This cascade is implemented in `validate_and_fill_config()`.

### BFCL Benchmark

Set `data.use_test_benchmark=true` and `data.test_split_name=bfcl_<category>` to run evaluation via the Berkeley Function Calling Leaderboard harness instead of the standard generate+score pipeline.

### Outputs

Experiments write to `outputs/YYYY-MM-DD/{strategy}_HH-MM-SS/iter_{n}/` containing `metrics.json`, `labeled_data.jsonl`, and model checkpoints. Pre-computed embeddings are cached in `cache/`.

### Dashboard

Streamlit app (`Welcome.py` + `pages/`) provides UI for experiment configuration, metric visualization, labeled example review, and manual annotation.

### Tests

`test/test_benchmark.py` uses `configs/test.yaml` (small dataset) for fast integration tests. Tests validate basic execution (`returncode=0`) and metric thresholds (`rouge1 >= 0.17`, `rougeL >= 0.13`). The test entry point is `HYDRA_CONFIG_NAME=test run-al +save_model=false`.
