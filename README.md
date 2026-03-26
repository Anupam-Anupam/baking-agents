# Baking Agents: WebArena Observer-Bake

This repository now tracks a focused goal:

1. Run an agent on WebArena shopping tasks.
2. Extract per-run observer feedback.
3. Distill feedback into stable teacher/student prompt recipes.
4. Bake those recipes with Bread.
5. Compare baked vs non-baked baselines on held-out validation tasks.

## What this currently accomplishes

- Bread SDK bake examples are still available in:
  - `example_bakes/example_yoda_bake.py`
  - `example_bakes/example_multi-target_bake.py`
- Utility scripts are still available in:
  - `helper_scripts/check_bake_status.py`
  - `helper_scripts/chat_with_model.py`
- This repo has been cleaned of older qubit-game evaluation/prototype files that were not aligned with the WebArena objective.

## How far we have gotten

- Architecture and workflow are defined:
  - trajectories -> observer lessons -> distilled rules -> teacher/student prompts -> stim/rollout -> bake -> ablation eval.
- Runtime blockers were identified in execution:
  - WebArena site stack must be live (shopping/admin + required domains).
  - Agent model endpoint must be reachable and OpenAI-compatible.
- Bread docs integration requirements are captured and validated at the workflow level, including stim/rollout ordering and target/bake semantics.

## What is still missing (the actual training)

The missing piece is full live training execution over real WebArena shopping runs:

- Start and verify all required WebArena services.
- Run a real 20-task train window with successful trajectories (not infra failures).
- Launch live Bread bake jobs (non-dry-run) from distilled rules.
- Run full held-out validation ablations and iterate on rule quality.

In short: the pipeline design is in place, but production-quality training data generation and live bake/eval cycles still need to run to completion.

## Setup

```bash
pip install -r requirements.txt
```

Set your Bread key:

```bash
export BREAD_API_KEY="your_key"
```

## References

- [Bread docs](https://docs.bread.com.ai)
- [Bread Stim API](https://docs.bread.com.ai/api-reference/targets-stim)
- [WebArena](https://github.com/web-arena-x/webarena)

## License

See `LICENSE`.