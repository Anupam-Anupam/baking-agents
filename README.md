# Baking Agents: WebArena Observer-Bake

This repo is now focused on one goal: **WebArena shopping-domain learning with Bread prompt baking**.

Pipeline:

1. Run WebArena shopping tasks with a base model.
2. Extract observer lessons from run logs.
3. Distill lessons into a teacher/student prompt recipe.
4. Run Bread stim -> rollout -> bake.
5. Evaluate baked vs non-baked baselines on held-out validation tasks.

## What is in the repo now

- WebArena-focused bake scaffold:
  - `src/webarena_bake/observer/`
  - `src/webarena_bake/memory/`
  - `src/webarena_bake/distillation/`
  - `src/webarena_bake/baking/`
  - `src/webarena_bake/runners/`
  - `src/webarena_bake/evaluation/`
- CLI entrypoints:
  - `scripts/build_shopping_split.py`
  - `scripts/train_loop.py`
  - `scripts/run_ablation.py`
- Runtime config:
  - `configs/default.json`
- Bread SDK examples and helper scripts:
  - `example_bakes/`
  - `helper_scripts/`

## What this already accomplishes

- Stable shopping-only split generation (80/20 by task id).
- Windowed training loop structure (20-task batches).
- Observer -> lesson store -> distill -> recipe generation.
- Bread target/stim/rollout/bake orchestration with dry-run support.
- Ablation runner for validation variants:
  - baseline
  - retrieval_only
  - distilled_prompt
  - baked
  - baked_plus_retrieval

## What is still missing (actual training)

Only live execution remains:

- Bring up full WebArena site stack (shopping/admin + required domains).
- Ensure model endpoints are reachable (policy + optional observer endpoint).
- Run non-dry Bread bakes (`dry_run_bake=false` or `--live-bake`).
- Iterate on real trajectories and run full held-out ablations.

In short: **code scaffold is in place; infrastructure-backed training/evaluation runs are the remaining step.**

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python scripts/build_shopping_split.py --config-dir /path/to/webarena/config_files
```

```bash
python scripts/train_loop.py --webarena-config-dir /path/to/webarena/config_files
```

```bash
python scripts/run_ablation.py --webarena-config-dir /path/to/webarena/config_files --baked-model-name your/repo/bake/checkpoint
```

## References

- [WebArena](https://github.com/web-arena-x/webarena)
- [Bread docs](https://docs.bread.com.ai)
- [Bread Stim API](https://docs.bread.com.ai/api-reference/targets-stim)