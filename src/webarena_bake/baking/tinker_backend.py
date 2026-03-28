from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from webarena_bake.schemas.types import PromptRecipe, WebArenaRunRecord
from webarena_bake.utils.hash import stable_hash
from webarena_bake.utils.io import write_json, write_jsonl


def _resolve_tinker_repo(workspace_dir: Path, tinker_repo_path: str | None) -> Path:
    if tinker_repo_path:
        candidate = Path(tinker_repo_path)
        if not candidate.is_absolute():
            candidate = (workspace_dir / candidate).resolve()
    else:
        candidate = (workspace_dir / "vendor" / "baking_with_tinker").resolve()
    if not (candidate / "bake.py").exists():
        raise FileNotFoundError(f"Tinker bake.py not found at {candidate}")
    return candidate


def _sync_tinker_env(repo_dir: Path) -> None:
    root_env = repo_dir / ".env"
    care_env = repo_dir / "care package" / ".env"
    if root_env.exists() and not care_env.exists():
        care_env.write_text(root_env.read_text(encoding="utf-8"), encoding="utf-8")
    elif care_env.exists() and not root_env.exists():
        root_env.write_text(care_env.read_text(encoding="utf-8"), encoding="utf-8")


def _build_sft_rows(recipe: PromptRecipe, run_records: list[WebArenaRunRecord], min_rows: int = 32) -> list[dict[str, Any]]:
    base_rules = recipe.teacher_prompt_text.strip()
    rows: list[dict[str, Any]] = []
    for record in run_records:
        user_prompt = (
            f"Task intent: {record.task_intent.strip() or 'N/A'}\n"
            f"Start URL: {record.start_url.strip() or 'N/A'}\n"
            "Provide the next best web-action policy for this task."
        )
        outcome = "success" if record.success else "failure"
        assistant_text = (
            f"Outcome observed: {outcome}.\n"
            "Behavior policy:\n"
            f"{base_rules}"
        )
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
        )

    # Keep the bake script stable even on tiny windows by padding examples.
    seed_row = rows[0] if rows else {
        "messages": [
            {"role": "user", "content": "Provide a robust web-navigation policy."},
            {"role": "assistant", "content": base_rules or "Prefer evidence-backed actions and stop safely."},
        ]
    }
    while len(rows) < min_rows:
        rows.append(seed_row)
    return rows


def _write_tinker_config(
    path: Path,
    tokenizer_model_name: str,
    train_base_model: str,
    log_dir: Path,
    num_epochs: int,
    top_k: int,
    batch_size: int,
) -> None:
    openrouter_model = tokenizer_model_name.lower()
    content = (
        '"""Auto-generated WebArena Tinker config."""\n\n'
        f'MODEL_NAME = "{tokenizer_model_name}"\n'
        f'BASE_MODEL_PATH = "{train_base_model}"\n'
        f'OPENROUTER_MODEL = "{openrouter_model}"\n'
        'RENDERER_NAME = "qwen3_disable_thinking"\n\n'
        "LORA_RANK = 32\n"
        f"TOP_K = {top_k}\n\n"
        "NUM_QUERIES = 200\n"
        "CONCURRENCY = 20\n"
        'OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"\n'
        "TEMPERATURE_DATA_GEN = 0.7\n"
        "MAX_TOKENS_RESPONSE = 512\n\n"
        f"BATCH_SIZE = {batch_size}\n"
        "LEARNING_RATE = 1e-4\n"
        f"NUM_EPOCHS = {num_epochs}\n"
        "MAX_LENGTH = 2048\n"
        "SAVE_EVERY = 20\n\n"
        "ADAM_BETA1 = 0.9\n"
        "ADAM_BETA2 = 0.95\n"
        "ADAM_EPS = 1e-8\n\n"
        "NUM_VERIFY_QUERIES = 10\n"
        "TEMPERATURE_VERIFY = 1.0\n"
        "MAX_TOKENS_VERIFY = 256\n\n"
        'PROMPT_FILE = "prompt.md"\n'
        'DATA_FILE = "baking_data.jsonl"\n'
        f'LOG_DIR = "{str(log_dir)}"\n\n'
        'WANDB_PROJECT = "webarena-bake"\n'
    )
    path.write_text(content, encoding="utf-8")


def _python_has_tinker(python_executable: str) -> bool:
    probe = subprocess.run(
        [python_executable, "-c", "import tinker, tinker_cookbook, wandb"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


def _latest_checkpoint_paths(log_dir: Path) -> dict[str, str]:
    checkpoints_path = log_dir / "checkpoints.jsonl"
    if not checkpoints_path.exists():
        return {}
    last_row: dict[str, Any] | None = None
    for line in checkpoints_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        last_row = row
    if not last_row:
        return {}
    result: dict[str, str] = {}
    for key in ("state_path", "sampler_path"):
        value = str(last_row.get(key, "")).strip()
        if value:
            result[key] = value
    return result


def run_tinker_window_bake(
    workspace_dir: Path,
    recipe: PromptRecipe,
    run_records: list[WebArenaRunRecord],
    bake_name: str,
    base_model: str | None,
    dry_run: bool,
    tinker_repo_path: str | None,
    tinker_python_executable: str,
    tinker_num_epochs: int,
    tinker_top_k: int,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_run_name: str | None,
    wandb_orchestrator_run_id: str | None,
    window_index: int | None,
    wandb_enable_child_run: bool,
) -> dict[str, Any]:
    repo_dir = _resolve_tinker_repo(workspace_dir, tinker_repo_path)
    _sync_tinker_env(repo_dir)
    out_dir = workspace_dir / "results" / "bake_eval" / recipe.recipe_version
    out_dir.mkdir(parents=True, exist_ok=True)
    training_base_model = (base_model or "Qwen/Qwen3-8B").strip()
    tokenizer_model_name = (
        os.environ.get("TINKER_TOKENIZER_MODEL", "Qwen/Qwen3-8B").strip()
        if training_base_model.startswith("tinker://")
        else training_base_model
    )

    prompt_path = repo_dir / "prompt.md"
    prompt_backup = prompt_path.with_suffix(".md.webarena_bak")
    if prompt_path.exists():
        shutil.copy2(prompt_path, prompt_backup)
    prompt_path.write_text(recipe.teacher_prompt_text.strip() + "\n", encoding="utf-8")

    data_rows = _build_sft_rows(recipe, run_records)
    write_jsonl(repo_dir / "baking_data.jsonl", data_rows)

    config_path = repo_dir / "config.py"
    config_backup = config_path.with_suffix(".py.webarena_bak")
    if config_path.exists():
        shutil.copy2(config_path, config_backup)
    _write_tinker_config(
        path=config_path,
        tokenizer_model_name=tokenizer_model_name,
        train_base_model=training_base_model,
        log_dir=out_dir / "tinker_logs",
        num_epochs=tinker_num_epochs,
        top_k=tinker_top_k,
        batch_size=max(4, min(16, len(data_rows) // 2)),
    )

    metadata = {
        "backend": "tinker",
        "bake_name": bake_name,
        "recipe_version": recipe.recipe_version,
        "repo_dir": str(repo_dir),
        "model_name": tokenizer_model_name,
        "base_model_used": training_base_model,
        "examples": len(data_rows),
        "dry_run": dry_run,
        "python_executable": tinker_python_executable,
        "recipe_hash": stable_hash(recipe.to_dict()),
        "data_hash": stable_hash(data_rows),
        "wandb_group": wandb_group or "",
        "wandb_run_name": wandb_run_name or "",
        "window_index": int(window_index or 0),
        "wandb_enable_child_run": bool(wandb_enable_child_run),
    }
    write_json(out_dir / "tinker_prep.json", metadata)

    if not _python_has_tinker(tinker_python_executable):
        result = {
            **metadata,
            "status": "blocked_missing_tinker_runtime",
            "hint": (
                f"Install runtime in {tinker_python_executable}: "
                f"pip install -e '{repo_dir / 'care package' / 'tinker-cookbook'}' wandb httpx python-dotenv"
            ),
        }
        write_json(out_dir / "bake_summary.json", result)
        return result

    if dry_run:
        result = {
            **metadata,
            "status": "dry_run_prepared",
            "command": f"{tinker_python_executable} bake.py",
        }
        write_json(out_dir / "bake_summary.json", result)
        return result

    env = os.environ.copy()
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    if wandb_group:
        env["WANDB_RUN_GROUP"] = wandb_group
    if wandb_run_name:
        env["WANDB_RUN_NAME"] = wandb_run_name
    if wandb_orchestrator_run_id:
        env["WANDB_ORCHESTRATOR_RUN_ID"] = wandb_orchestrator_run_id
    if window_index is not None:
        env["WANDB_WINDOW_INDEX"] = str(window_index)
    env.setdefault("WANDB_JOB_TYPE", "window_bake")
    if not wandb_enable_child_run:
        env["WANDB_MODE"] = "disabled"
    train_metrics_path = out_dir / "tinker_train_metrics.json"
    env["TINKER_TRAIN_METRICS_PATH"] = str(train_metrics_path)
    command = [tinker_python_executable, "bake.py"]
    proc = subprocess.run(command, cwd=repo_dir, env=env, capture_output=True, text=True, check=False)
    (out_dir / "tinker_bake_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "tinker_bake_stderr.log").write_text(proc.stderr, encoding="utf-8")
    lower_err = proc.stderr.lower()
    failure_reason = ""
    if "error code: 402" in lower_err and "billing" in lower_err:
        failure_reason = "tinker_billing_blocked"

    checkpoint_paths = _latest_checkpoint_paths(out_dir / "tinker_logs")
    train_metrics: dict[str, Any] = {}
    if train_metrics_path.exists():
        try:
            train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            train_metrics = {}
    result = {
        **metadata,
        "status": "completed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "failure_reason": failure_reason,
        "tinker_state_path": checkpoint_paths.get("state_path", ""),
        "tinker_sampler_path": checkpoint_paths.get("sampler_path", ""),
        "train_metrics": train_metrics,
        "stdout_log": str(out_dir / "tinker_bake_stdout.log"),
        "stderr_log": str(out_dir / "tinker_bake_stderr.log"),
    }
    write_json(out_dir / "bake_summary.json", result)
    return result
