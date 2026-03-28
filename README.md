# WebArena Bake Loop (Tinker Backend)

End-to-end pipeline for iterative WebArena training:

1. Run WebArena tasks.
2. Extract observer lessons from each run.
3. Distill lessons into a domain prompt recipe.
4. Bake with `baking_with_tinker`.
5. Repeat in windows (`20 -> bake -> next 20 -> bake`) with checkpoint chaining.

This README is a practical runbook for running your own setup.

## 0) Fastest Way (One Command)

If you want the full intended workflow with minimal setup commands, use:

```bash
./scripts/run_full_training.sh
```

That script:

1. runs preflight (`doctor.py`)
2. prints planned number of bake windows
3. starts the full train loop with Tinker live bakes

Dry-run mode:

```bash
./scripts/run_full_training.sh --dry-run
```

## 1) Repository Assumptions

This repo expects a sibling `webarena` checkout:

- `../webarena` must exist relative to `webarena-bake`
- WebArena task configs are in `../webarena/config_files`
- Default WebArena python is `../webarena/.conda-py310/bin/python`

If your paths differ, update `configs/default.json`.
In particular, set `webarena_python_executable` to a valid interpreter for your machine.

## 1.1) Apply Required WebArena Patch (Important)

This project depends on a small set of WebArena-side code updates (provider wiring, evaluator model config, auth/login robustness).

Apply them once:

```bash
./scripts/apply_webarena_patch.sh
```

If your WebArena checkout is not at `../webarena`:

```bash
./scripts/apply_webarena_patch.sh /absolute/path/to/webarena
```

Patch source tracked in this repo:

- `patches/webarena_required.patch`

## 1.2) Prepare WebArena Runtime (Required)

Before building splits or running preflight, make sure your sibling `webarena` checkout is runnable.

From `../webarena`:

```bash
pip install -e .
```

If you run into missing-module errors in `webarena` scripts, install WebArena runtime deps:

```bash
pip install -r requirements.txt
```

Then install Playwright browser binaries (required for `run.py`/smoke gates):

```bash
python3 -m playwright install chromium
```

> Note: WebArena dependency resolution can be Python-version sensitive on some systems.
> If `pip install -r requirements.txt` fails in your default interpreter, use a Python version/environment supported by your WebArena checkout.

## 2) Install Dependencies

From `webarena-bake`:

```bash
pip install -r requirements.txt
pip install -e "vendor/baking_with_tinker/care package/tinker-cookbook"
```

## 3) Set Environment Variables

Create `../.env` (project root) with at least:

- `OPENAI_API_KEY`
- `TINKER_API_KEY`
- `WANDB_API_KEY`

The train script loads `.env` from either:

- `webarena-bake/.env`
- `../.env`

For the vendored Tinker repo, credentials are mirrored to:

- `vendor/baking_with_tinker/.env`
- `vendor/baking_with_tinker/care package/.env`

## 4) Start Local Model Endpoints (OpenAI-compatible)

Default config points to:

- policy: `http://localhost:8000/v1`
- observer: `http://localhost:8001/v1`

Recommended for real training (policy endpoint that can serve base + baked checkpoints):

```bash
python3 scripts/tinker_openai_server.py --port 8000
```

Observer endpoint can be either your own LLM server or a lightweight mock:

```bash
python3 scripts/mock_openai_server.py --port 8001
```

Important: do not use `scripts/mock_openai_server.py --port 8000` for policy training.
That server always returns a canned `stop [N/A]` action and will cause near-all task failures.

## 5) Build Task Splits

First generate WebArena task configs (one-time, or whenever site URLs change):

```bash
cd ../webarena
python3 scripts/generate_test_data.py
cd -
```

Generate shopping train/val split from WebArena configs:

```bash
python3 scripts/build_shopping_split.py \
  --config-dir "../webarena/config_files" \
  --output-dir "./data/splits"
```

Default split file used by training:

- `data/splits/shopping_split_manifest.json`

## 6) Run Preflight Checks

Run diagnostics before training:

```bash
python3 scripts/doctor.py --webarena-config-dir "../webarena/config_files"
```

Optional infra scripts:

- `python3 scripts/bringup_infra.py`
- `python3 scripts/bootstrap_auth.py --webarena-config-dir "../webarena/config_files"`
- `python3 scripts/smoke_gates.py --webarena-config-dir "../webarena/config_files"`

Important notes:

- `bringup_infra.py` only starts containers named `shopping`, `shopping_admin`, `forum`, and `gitlab` if they already exist on your machine.
- If those containers do not exist yet, follow WebArena's `environment_docker/README.md` to create them, or use reachable hosted site URLs.

## 7) Configure Training

Main config: `configs/default.json`

Important defaults:

- `bake_backend: "tinker"`
- `batch_size: 20`
- `dry_run_bake: true` (set live via `--live-bake`)

Common fields to edit:

- `policy_model_endpoint`
- `observer_model_endpoint`
- `policy_model_name`
- `observer_model_name`
- `webarena_root`
- `webarena_python_executable`
- `tinker_num_epochs`
- `tinker_top_k`

## 8) Run Training

### Dry-run bake preparation

```bash
python3 scripts/train_loop.py \
  --webarena-config-dir "../webarena/config_files" \
  --bake-backend tinker
```

### Live baking

```bash
python3 scripts/train_loop.py \
  --webarena-config-dir "../webarena/config_files" \
  --bake-backend tinker \
  --live-bake
```

### Full dataset (default split) in one command

```bash
./scripts/run_full_training.sh
```

### How many bakes will run?

Number of bakes = `ceil(train_count / batch_size)`.

With current defaults:

- `train_count = 312`
- `batch_size = 20`
- planned bakes = `16`

## 9) What Happens Per Window

For each window (`batch_size` tasks):

1. WebArena run records:
   - `results/runs/window_XXX/batch_summary.json`
2. Observer lessons:
   - `results/observer/window_XXX/lessons.jsonl`
3. Distillation outputs:
   - `results/distill/window_XXX/distilled_rules.json`
   - `results/distill/window_XXX/bread_recipe.json`
4. Tinker bake prep + execution:
   - `results/bake_eval/window_XXX/tinker_prep.json`
   - `results/bake_eval/window_XXX/bake_summary.json`
   - `results/bake_eval/window_XXX/tinker_bake_stdout.log`
   - `results/bake_eval/window_XXX/tinker_bake_stderr.log`

## 10) Checkpoint Chaining (v1 -> v2 -> v3)

The pipeline now chains Tinker checkpoints across windows:

- window 1 bakes from base model
- window 2 bakes from window 1 state checkpoint
- window 3 bakes from window 2 state checkpoint

Lineage file:

- `results/state/lineage.json`

For each window, verify:

- `bake_base_model` (input state used for bake)
- `tinker_state_path` (new train state)
- `tinker_sampler_path` (new serving path)

If chaining is correct, window N+1 `bake_base_model` should equal window N `tinker_state_path`.

## 10.1) W&B Behavior (Single Training Run)

Default behavior is now one online W&B run per full training session:

- `wandb_single_run: true`
- `wandb_enable_child_runs: false`

This means all windows/bakes log into one run as steps (`window_index`), including training metrics from each bake:

- `window/train_last_avg_kl_per_token`
- `window/train_last_kl_loss`
- `window/train_last_lr`
- `window/train_last_datums`
- `window/train_steps`

If you want the old behavior (separate child bake runs), set:

- `wandb_enable_child_runs: true`

## 11) Run Your Own Domain

To run a different task/domain:

1. Point `--webarena-config-dir` to your task configs.
2. Build a split with `scripts/build_shopping_split.py` or provide your own split manifest with:
   - `train_ids`
   - `val_ids`
3. Set domain-specific prompts/instructions via `webarena_instruction_path`.
4. Train with `scripts/train_loop.py --split-manifest <your_manifest>`.

The observer/distiller output rules and bake data from whatever tasks are in your manifest.

## 12) Validation / Ablation

Run evaluation variants:

```bash
python3 scripts/run_ablation.py \
  --webarena-config-dir "../webarena/config_files" \
  --baked-model-name "<your-baked-model>"
```

## 13) Troubleshooting

- **`OPENAI_API_KEY` missing**
  - Ensure key is in `../.env` or `webarena-bake/.env`.
- **Tinker billing / access errors**
  - Check Tinker console billing and API key.
- **WebArena evaluator model errors**
  - `webarena/evaluation_harness/helper_functions.py` uses `OPENAI_EVAL_MODEL` (default `gpt-4o-mini`).
- **NLTK `punkt_tab` errors**
  - Install in WebArena python env:
    ```bash
    "../webarena/.conda-py310/bin/python" -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
    ```

## 14) Key Scripts

- `scripts/build_shopping_split.py`
- `scripts/train_loop.py`
- `scripts/run_full_training.sh`
- `scripts/run_ablation.py`
- `scripts/doctor.py`
- `scripts/bringup_infra.py`
- `scripts/bootstrap_auth.py`
- `scripts/smoke_gates.py`
