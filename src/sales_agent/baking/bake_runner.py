from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sales_agent.baking.baked_model_registry import BakedModelRegistry
from sales_agent.baking.bread_client import BreadClient
from sales_agent.baking.rollout_prep import build_rollout_generators
from sales_agent.baking.stim_generation import generate_stim_jsonl
from sales_agent.baking.target_builder import TargetSpec
from sales_agent.schemas.types import BakeLineageRecord, PromptRecipe
from sales_agent.utils.hash import stable_hash
from sales_agent.utils.io import write_json


@dataclass
class BakeRunConfig:
    repo_name: str
    base_model: str | None
    target_name: str
    bake_name: str
    stim_count: int = 50
    dry_run: bool = True


def run_bake_pipeline(
    recipe: PromptRecipe,
    target_spec: TargetSpec,
    config: BakeRunConfig,
    workspace_dir: Path,
) -> dict:
    bread = BreadClient(dry_run=config.dry_run)
    out_dir = workspace_dir / "results" / "bake_eval" / recipe.recipe_version
    out_dir.mkdir(parents=True, exist_ok=True)

    bread.repo_set(repo_name=config.repo_name, base_model=config.base_model)
    bread.prompt_set(config.repo_name, recipe.teacher_prompt_name, recipe.teacher_prompt_text)
    bread.prompt_set(config.repo_name, recipe.student_prompt_name, recipe.student_prompt_text)

    stim_path = workspace_dir / "data" / "bread" / "stim" / f"{target_spec.task_family}_{recipe.recipe_version}.jsonl"
    stim_rows = generate_stim_jsonl(target_spec.task_family, stim_path, count=config.stim_count)
    generators = build_rollout_generators(stim_rows)
    bread.target_set(
        repo_name=config.repo_name,
        target_name=config.target_name,
        teacher_prompt_name=recipe.teacher_prompt_name,
        student_prompt_name=recipe.student_prompt_name,
        generators=generators,
    )

    stim_result = bread.run_stim(config.repo_name, config.target_name)
    rollout_result = bread.run_rollout(config.repo_name, config.target_name)
    bread.bake_set(config.repo_name, config.bake_name, config.target_name)
    bake_result = bread.bake_run(config.repo_name, config.bake_name)

    recipe_hash = stable_hash(recipe.to_dict())
    stim_hash = stable_hash(stim_rows)
    rollout_hash = stable_hash(generators)

    registry = BakedModelRegistry(workspace_dir / "data" / "bread" / "baked_models" / "registry.json")
    registry.add_record(
        BakeLineageRecord(
            bake_id=str(bake_result.get("bake_id", config.bake_name)),
            repo_name=config.repo_name,
            base_model=config.base_model or "default",
            teacher_prompt_version=recipe.recipe_version,
            stim_version=f"{target_spec.task_family}_{recipe.recipe_version}",
            rollout_version=recipe.recipe_version,
            recipe_hash=recipe_hash,
            stim_hash=stim_hash,
            rollout_hash=rollout_hash,
            notes="Generated from distilled rules pipeline",
        )
    )

    summary = {
        "repo_name": config.repo_name,
        "target_name": config.target_name,
        "bake_name": config.bake_name,
        "stim_result": stim_result,
        "rollout_result": rollout_result,
        "bake_result": bake_result,
        "dry_run": config.dry_run,
    }
    write_json(out_dir / "bake_summary.json", summary)
    return summary

