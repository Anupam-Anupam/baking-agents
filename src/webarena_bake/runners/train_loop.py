from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path

from webarena_bake.baking.bake_runner import BakeConfig, run_window_bake
from webarena_bake.baking.target_builder import TargetSpec
from webarena_bake.distillation.distill_window import distill_window
from webarena_bake.memory.lesson_store import LessonStore
from webarena_bake.observer.episode_to_lessons import extract_lessons
from webarena_bake.runners.preflight import run_preflight
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
    run_preflight_checks: bool = True
    bake_backend: str = "bread_sdk"
    tinker_repo_path: str | None = None
    tinker_python_executable: str = "python3"
    tinker_num_epochs: int = 1
    tinker_top_k: int = 20
    wandb_project: str = "webarena-bake"
    wandb_entity: str | None = None
    wandb_single_run: bool = True
    wandb_run_name: str | None = None
    wandb_enable_child_runs: bool = False


def _runtime_metadata(config: LoopConfig, split: dict, current_model: str) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": config.provider,
        "policy_model": current_model,
        "policy_endpoint": config.model_endpoint,
        "observer_endpoint": config.observer_endpoint,
        "observer_model": config.observer_model,
        "bake_backend": config.bake_backend,
        "tinker_repo_path": config.tinker_repo_path or "",
        "tinker_python_executable": config.tinker_python_executable,
        "wandb_project": config.wandb_project,
        "wandb_entity": config.wandb_entity or "",
        "wandb_single_run": config.wandb_single_run,
        "wandb_run_name": config.wandb_run_name or "",
        "wandb_enable_child_runs": config.wandb_enable_child_runs,
        "batch_size": config.batch_size,
        "split_seed": split.get("seed"),
        "train_ratio": split.get("train_ratio"),
        "train_count": len(split.get("train_ids", [])),
        "val_count": len(split.get("val_ids", [])),
        "webarena_root": str(config.webarena_root),
        "webarena_config_dir": str(config.webarena_config_dir),
        "webarena_python_executable": config.webarena_python_executable,
        "env": {name: os.environ.get(name, "") for name in ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "MAP", "WIKIPEDIA", "HOMEPAGE")},
    }


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_train_loop(workspace_dir: Path, config: LoopConfig) -> dict:
    split = read_json(config.split_manifest_path)
    train_ids = list(split.get("train_ids", []))
    windows = _chunked(train_ids, config.batch_size)

    state_dir = workspace_dir / "results" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    current_model = config.initial_model_name
    current_bake_base_model = (
        (config.bread_base_model or config.initial_model_name) if config.bake_backend == "tinker" else config.bread_base_model
    )
    lineage: list[dict] = []
    orchestrator_wandb = None
    wandb_group = ""
    orchestrator_run_id = ""

    if config.wandb_single_run and os.environ.get("WANDB_API_KEY", ""):
        try:
            import wandb

            run_name = config.wandb_run_name or f"webarena-train-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            orchestrator_wandb = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=run_name,
                job_type="orchestrator",
                config={
                    "provider": config.provider,
                    "initial_model_name": config.initial_model_name,
                    "batch_size": config.batch_size,
                    "bake_backend": config.bake_backend,
                    "train_windows": len(windows),
                },
            )
            if orchestrator_wandb is not None:
                orchestrator_run_id = str(getattr(orchestrator_wandb, "id", "") or "")
                wandb_group = orchestrator_run_id or run_name
        except Exception:
            orchestrator_wandb = None

    try:
        if config.run_preflight_checks:
            preflight = run_preflight(
                webarena_root=config.webarena_root,
                webarena_config_dir=config.webarena_config_dir,
                task_ids=train_ids[: min(len(train_ids), config.batch_size)],
                policy_model_endpoint=config.model_endpoint,
                observer_model_endpoint=config.observer_endpoint,
                provider=config.provider,
            )
            write_json(state_dir / "preflight.json", preflight)
            if not preflight.get("ok", False):
                raise RuntimeError(
                    "Preflight checks failed. See results/state/preflight.json for details and remediation hints."
                )

        write_json(state_dir / "run_metadata.json", _runtime_metadata(config, split, current_model))

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
            window_observer_dir = workspace_dir / "results" / "observer" / run_id
            window_observer_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(window_observer_dir / "lessons.jsonl", (item.to_dict() for item in lessons))

            note_store = LessonStore(workspace_dir / "results" / "observer" / "window_notes.jsonl")
            note_store.clear()
            note_store.add_lessons(lessons)
            distill_dir = workspace_dir / "results" / "distill" / run_id
            recipe = distill_window(
                lessons=note_store.all_lessons(),
                output_dir=distill_dir,
                recipe_version=run_id,
                min_support=2,
                min_confidence=0.65,
                max_contradictions=0,
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
                    base_model=current_bake_base_model,
                    bake_name=f"shopping_bake_{run_id}",
                    target_name=f"shopping_target_{run_id}",
                    dry_run=config.dry_run_bake,
                    stim_count=max(100, len(task_ids) * 8),
                    backend=config.bake_backend,
                    tinker_repo_path=config.tinker_repo_path,
                    tinker_python_executable=config.tinker_python_executable,
                    tinker_num_epochs=config.tinker_num_epochs,
                    tinker_top_k=config.tinker_top_k,
                    wandb_project=config.wandb_project,
                    wandb_entity=config.wandb_entity,
                    wandb_group=wandb_group,
                    wandb_run_name=f"{(config.wandb_run_name or 'webarena-train')}-{run_id}",
                    wandb_orchestrator_run_id=orchestrator_run_id,
                    window_index=window_index,
                    wandb_enable_child_run=config.wandb_enable_child_runs,
                ),
                run_records=run_records,
            )
            bake_status = str(bake_summary.get("status", "")).strip().lower()
            if bake_status and bake_status not in {"completed", "dry_run_prepared"}:
                failure_reason = str(bake_summary.get("failure_reason", "")).strip()
                raise RuntimeError(
                    f"Bake failed in {run_id} with status={bake_status}"
                    + (f", reason={failure_reason}" if failure_reason else "")
                    + ". Check results/bake_eval/<run_id>/bake_summary.json and Tinker logs."
                )

            tinker_sampler_path = str(bake_summary.get("tinker_sampler_path", "")).strip()
            tinker_state_path = str(bake_summary.get("tinker_state_path", "")).strip()
            if config.bake_backend == "tinker" and tinker_sampler_path:
                baked_model = tinker_sampler_path
            else:
                baked_model = f"{config.bread_repo_name}/{bake_summary.get('bake_name', f'shopping_bake_{run_id}')}/latest"
            current_model = baked_model
            if config.bake_backend == "tinker" and tinker_state_path:
                current_bake_base_model = tinker_state_path
            lineage_item = {
                "window": window_index,
                "run_id": run_id,
                "input_model": run_records[0].model_name if run_records else "",
                "output_model": baked_model,
                "bake_base_model": current_bake_base_model if config.bake_backend != "tinker" else str(
                    bake_summary.get("base_model_used", "")
                ),
                "tinker_state_path": tinker_state_path,
                "tinker_sampler_path": tinker_sampler_path,
                "task_ids": task_ids,
                "bake_id": bake_summary.get("bake_id", ""),
                "recipe_version": recipe.recipe_version,
            }
            lineage.append(lineage_item)
            write_json(state_dir / "lineage.json", {"lineage": lineage, "current_model": current_model})

            if orchestrator_wandb is not None:
                pass_count = sum(1 for item in run_records if item.success)
                avg_score = sum(item.score for item in run_records) / max(len(run_records), 1)
                train_metrics = dict(bake_summary.get("train_metrics", {}) or {})
                train_logs = {}
                for key in ("last_avg_kl_per_token", "last_kl_loss", "last_lr", "last_datums", "steps"):
                    if key in train_metrics:
                        train_logs[f"window/train_{key}"] = float(train_metrics[key])
                orchestrator_wandb.log(
                    {
                        "window_index": window_index,
                        "window/pass_count": pass_count,
                        "window/task_count": len(run_records),
                        "window/pass_rate": pass_count / max(len(run_records), 1),
                        "window/avg_score": avg_score,
                        "window/bake_status": 1.0 if bake_status == "completed" else 0.0,
                        **train_logs,
                    },
                    step=window_index,
                )
    finally:
        if orchestrator_wandb is not None:
            try:
                orchestrator_wandb.finish()
            except Exception:
                pass

    return {"windows": len(windows), "final_model": current_model, "lineage": lineage}

