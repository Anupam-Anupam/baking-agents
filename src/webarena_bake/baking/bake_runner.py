from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from webarena_bake.baking.baked_model_registry import BakedModelRegistry
from webarena_bake.baking.bread_client import BreadClient
from webarena_bake.baking.rollout_prep import build_generators
from webarena_bake.baking.stim_generation import generate_stim_jsonl
from webarena_bake.baking.target_builder import TargetSpec
from webarena_bake.schemas.types import BakeLineageRecord, PromptRecipe
from webarena_bake.utils.hash import stable_hash
from webarena_bake.utils.io import write_json


@dataclass
class BakeConfig:
    repo_name: str
    base_model: str | None
    bake_name: str
    target_name: str
    dry_run: bool = True
    stim_count: int = 200


def run_window_bake(workspace_dir: Path, recipe: PromptRecipe, target_spec: TargetSpec, config: BakeConfig) -> dict[str, Any]:
    client = BreadClient(dry_run=config.dry_run)
    out_dir = workspace_dir / "results" / "bake_eval" / recipe.recipe_version
    out_dir.mkdir(parents=True, exist_ok=True)

    client.repo_set(config.repo_name, config.base_model)
    client.prompt_set(config.repo_name, recipe.teacher_prompt_name, recipe.teacher_prompt_text)
    client.prompt_set(config.repo_name, recipe.student_prompt_name, recipe.student_prompt_text)

    stim_path = workspace_dir / "data" / "bread" / "stim" / f"{recipe.recipe_version}.jsonl"
    stim_rows = generate_stim_jsonl(stim_path, count=config.stim_count)
    generators = build_generators(stim_rows)

    client.target_set(
        repo_name=config.repo_name,
        target_name=config.target_name,
        teacher_prompt_name=target_spec.teacher_prompt_name,
        student_prompt_name=target_spec.student_prompt_name,
        generators=generators,
    )
    stim_result = client.run_stim(config.repo_name, config.target_name)
    rollout_result = client.run_rollout(config.repo_name, config.target_name)
    client.bake_set(config.repo_name, config.bake_name, config.target_name)
    bake_result = client.bake_run(config.repo_name, config.bake_name)

    bake_id = str(bake_result.get("bake_id", config.bake_name))
    registry = BakedModelRegistry(workspace_dir / "data" / "bread" / "baked_models" / "registry.json")
    registry.add_record(
        BakeLineageRecord(
            bake_id=bake_id,
            repo_name=config.repo_name,
            base_model=config.base_model or "default",
            teacher_prompt_version=recipe.recipe_version,
            stim_version=recipe.recipe_version,
            rollout_version=recipe.recipe_version,
            recipe_hash=stable_hash(recipe.to_dict()),
            stim_hash=stable_hash(stim_rows),
            rollout_hash=stable_hash(generators),
            notes="Window bake from observer distilled rules",
        )
    )

    summary = {
        "recipe_version": recipe.recipe_version,
        "repo_name": config.repo_name,
        "target_name": config.target_name,
        "bake_name": config.bake_name,
        "bake_id": bake_id,
        "dry_run": config.dry_run,
        "stim_result": stim_result,
        "rollout_result": rollout_result,
        "bake_result": bake_result,
    }
    write_json(out_dir / "bake_summary.json", summary)
    return summary

