from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from webarena_bake.baking.bake_runner import BakeConfig, run_window_bake
from webarena_bake.baking.target_builder import TargetSpec
from webarena_bake.distillation.distill_window import distill_window
from webarena_bake.memory.lesson_store import LessonStore
from webarena_bake.observer.episode_to_lessons import extract_lessons
from webarena_bake.runners.webarena_batch_runner import run_batch
from webarena_bake.utils.io import read_json, write_json, write_jsonl


@dataclass
class LoopConfig:
    webarena_root: Path
    webarena_config_dir: Path
    split_manifest_path: Path
    instruction_path: str
    provider: str
    initial_model_name: str
    model_endpoint: str
    batch_size: int
    bread_repo_name: str
    bread_base_model: str | None
    dry_run_bake: bool
    observer_endpoint: str | None = None
    observer_model: str | None = None
    observer_api_key: str | None = None
    webarena_python_executable: str = "python"


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_train_loop(workspace_dir: Path, config: LoopConfig) -> dict:
    split = read_json(config.split_manifest_path)
    train_ids = list(split.get("train_ids", []))
    windows = _chunked(train_ids, config.batch_size)

    current_model = config.initial_model_name
    lineage: list[dict] = []
    state_dir = workspace_dir / "results" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    for window_index, task_ids in enumerate(windows, start=1):
        run_id = f"window_{window_index:03d}"
        run_records = run_batch(
            webarena_root=config.webarena_root,
            config_dir=config.webarena_config_dir,
            task_ids=task_ids,
            run_id=run_id,
            instruction_path=config.instruction_path,
            provider=config.provider,
            model_name=current_model,
            model_endpoint=config.model_endpoint,
            result_dir=workspace_dir / "results" / "runs",
            python_executable=config.webarena_python_executable,
        )

        lessons = extract_lessons(
            run_records,
            task_family="shopping",
            observer_endpoint=config.observer_endpoint,
            observer_model=config.observer_model,
            observer_api_key=config.observer_api_key,
        )
        obs_dir = workspace_dir / "results" / "observer" / run_id
        obs_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(obs_dir / "lessons.jsonl", (l.to_dict() for l in lessons))

        notes = LessonStore(workspace_dir / "results" / "observer" / "window_notes.jsonl")
        notes.clear()
        notes.add_lessons(lessons)

        recipe = distill_window(
            lessons=notes.all_lessons(),
            output_dir=workspace_dir / "results" / "distill" / run_id,
            recipe_version=run_id,
        )
        bake_summary = run_window_bake(
            workspace_dir=workspace_dir,
            recipe=recipe,
            target_spec=TargetSpec(
                repo_name=config.bread_repo_name,
                target_name=f"shopping_target_{run_id}",
                teacher_prompt_name=recipe.teacher_prompt_name,
                student_prompt_name=recipe.student_prompt_name,
            ),
            config=BakeConfig(
                repo_name=config.bread_repo_name,
                base_model=config.bread_base_model,
                bake_name=f"shopping_bake_{run_id}",
                target_name=f"shopping_target_{run_id}",
                dry_run=config.dry_run_bake,
                stim_count=max(100, len(task_ids) * 8),
            ),
        )

        current_model = f"{config.bread_repo_name}/{bake_summary['bake_name']}/latest"
        lineage.append(
            {
                "window": window_index,
                "run_id": run_id,
                "input_model": run_records[0].model_name if run_records else "",
                "output_model": current_model,
                "task_ids": task_ids,
                "bake_id": bake_summary.get("bake_id", ""),
                "recipe_version": recipe.recipe_version,
            }
        )
        write_json(state_dir / "lineage.json", {"lineage": lineage, "current_model": current_model})

    return {"windows": len(windows), "final_model": current_model, "lineage": lineage}

