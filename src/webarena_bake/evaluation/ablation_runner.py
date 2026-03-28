from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from webarena_bake.evaluation.ablation_matrix import ABLATION_VARIANTS
from webarena_bake.runners.preflight import run_preflight
from webarena_bake.runners.webarena_batch_runner import run_batch
from webarena_bake.utils.io import read_json, write_json


@dataclass
class AblationConfig:
    webarena_root: Path
    webarena_config_dir: Path
    split_manifest_path: Path
    instruction_path: str
    provider: str
    base_model_name: str
    baked_model_name: str
    model_endpoint: str
    max_tasks: int | None = None
    webarena_python_executable: str = "python"
    observer_endpoint: str | None = None
    run_preflight_checks: bool = True


def _variant_model_name(variant_name: str, base_model_name: str, baked_model_name: str) -> str:
    if variant_name in {"baked", "baked_plus_retrieval"}:
        return baked_model_name
    return base_model_name


def run_ablation(workspace_dir: Path, config: AblationConfig) -> dict:
    split = read_json(config.split_manifest_path)
    val_ids = list(split.get("val_ids", []))
    if config.max_tasks is not None:
        val_ids = val_ids[: config.max_tasks]

    if config.run_preflight_checks:
        preflight = run_preflight(
            webarena_root=config.webarena_root,
            webarena_config_dir=config.webarena_config_dir,
            task_ids=val_ids[: min(len(val_ids), 5)],
            policy_model_endpoint=config.model_endpoint,
            observer_model_endpoint=config.observer_endpoint,
            provider=config.provider,
        )
        write_json(workspace_dir / "results" / "eval" / "preflight.json", preflight)
        if not preflight.get("ok", False):
            raise RuntimeError(
                "Ablation preflight failed. See results/eval/preflight.json for details and remediation hints."
            )

    summary: dict[str, dict] = {}
    for variant in ABLATION_VARIANTS:
        run_id = f"ablation_{variant.name}"
        records = run_batch(
            webarena_root=config.webarena_root,
            config_dir=config.webarena_config_dir,
            task_ids=val_ids,
            run_id=run_id,
            instruction_path=config.instruction_path,
            provider=config.provider,
            model_name=_variant_model_name(variant.name, config.base_model_name, config.baked_model_name),
            model_endpoint=config.model_endpoint,
            result_dir=workspace_dir / "results" / "eval" / "runs",
            python_executable=config.webarena_python_executable,
        )
        total = len(records)
        success = sum(1 for item in records if item.success)
        avg_score = (sum(item.score for item in records) / total) if total else 0.0
        summary[variant.name] = {
            "description": variant.description,
            "num_tasks": total,
            "success_rate": (success / total) if total else 0.0,
            "avg_score": avg_score,
        }

    payload = {"variants": summary, "val_count": len(val_ids)}
    write_json(workspace_dir / "results" / "eval" / "summary.json", payload)
    return payload

