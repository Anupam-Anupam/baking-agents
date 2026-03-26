from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from webarena_bake.schemas.types import WebArenaRunRecord
from webarena_bake.utils.io import read_json, write_json, write_jsonl


def _load_task_metadata(config_file: Path) -> dict[str, Any]:
    data = read_json(config_file)
    return {
        "task_id": str(data.get("task_id", config_file.stem)),
        "intent": str(data.get("intent", "")),
        "start_url": str(data.get("start_url", "")),
    }


def run_task(
    webarena_root: Path,
    result_dir: Path,
    instruction_path: str,
    provider: str,
    model_name: str,
    model_endpoint: str,
    config_file: Path,
    run_id: str,
    python_executable: str = "python",
) -> WebArenaRunRecord:
    meta = _load_task_metadata(config_file)
    task_out_dir = result_dir / run_id / meta["task_id"]
    task_out_dir.mkdir(parents=True, exist_ok=True)

    start_idx = config_file.stem
    try:
        end_idx = str(int(config_file.stem) + 1)
    except ValueError:
        end_idx = config_file.stem

    command = [
        python_executable,
        "run.py",
        "--instruction_path",
        instruction_path,
        "--test_start_idx",
        start_idx,
        "--test_end_idx",
        end_idx,
        "--provider",
        provider,
        "--model",
        model_name,
        "--model_endpoint",
        model_endpoint,
        "--result_dir",
        str(task_out_dir),
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{webarena_root}:{existing_pythonpath}" if existing_pythonpath else str(webarena_root)
    proc = subprocess.run(command, cwd=webarena_root, capture_output=True, text=True, env=env)
    success = proc.returncode == 0

    record = WebArenaRunRecord(
        run_id=run_id,
        task_id=meta["task_id"],
        config_file=str(config_file),
        task_intent=meta["intent"],
        start_url=meta["start_url"],
        model_name=model_name,
        success=success,
        score=0.75 if success else 0.0,
        steps=0,
        terminated=True,
        error="" if success else proc.stderr[-4000:],
        artifacts={
            "result_dir": str(task_out_dir),
            "stdout_file": str(task_out_dir / "stdout.log"),
            "stderr_file": str(task_out_dir / "stderr.log"),
        },
        raw={"returncode": proc.returncode},
    )
    (task_out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (task_out_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    write_json(task_out_dir / "run_record.json", record.to_dict())
    return record


def run_batch(
    webarena_root: Path,
    config_dir: Path,
    task_ids: list[str],
    run_id: str,
    instruction_path: str,
    provider: str,
    model_name: str,
    model_endpoint: str,
    result_dir: Path,
    python_executable: str = "python",
) -> list[WebArenaRunRecord]:
    records: list[WebArenaRunRecord] = []
    for task_id in task_ids:
        config_file = config_dir / f"{task_id}.json"
        if not config_file.exists():
            records.append(
                WebArenaRunRecord(
                    run_id=run_id,
                    task_id=task_id,
                    config_file=str(config_file),
                    task_intent="",
                    start_url="",
                    model_name=model_name,
                    success=False,
                    score=0.0,
                    steps=0,
                    terminated=True,
                    error="config_file_missing",
                )
            )
            continue
        records.append(
            run_task(
                webarena_root=webarena_root,
                result_dir=result_dir,
                instruction_path=instruction_path,
                provider=provider,
                model_name=model_name,
                model_endpoint=model_endpoint,
                config_file=config_file,
                run_id=run_id,
                python_executable=python_executable,
            )
        )

    write_json(result_dir / run_id / "batch_summary.json", {"run_id": run_id, "records": [r.to_dict() for r in records]})
    write_jsonl(result_dir / run_id / "batch_records.jsonl", (r.to_dict() for r in records))
    return records


def load_batch_records(batch_dir: Path) -> list[WebArenaRunRecord]:
    path = batch_dir / "batch_summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [WebArenaRunRecord(**item) for item in payload.get("records", [])]

