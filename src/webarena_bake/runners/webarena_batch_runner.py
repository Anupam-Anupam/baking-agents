from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from webarena_bake.runners.preflight import DEFAULT_SITE_ENV
from webarena_bake.schemas.types import WebArenaRunRecord
from webarena_bake.utils.io import read_json, write_json, write_jsonl

def _classify_error(stdout: str, stderr: str, return_code: int) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "openai_api_key environment variable must be set" in text:
        return "llm_auth_missing"
    if "err_connection_refused" in text or "failed to connect to localhost" in text:
        return "infra_service_unreachable"
    if "storage_state" in text and "assertionerror" in text:
        return "auth_storage_state_missing"
    if "auto_login.py" in text and "locator.fill: timeout" in text:
        return "auth_login_failed"
    if "no task left to run" in text:
        return "webarena_no_task_selected"
    if "[result] (fail)" in text:
        return "task_failed"
    if return_code == 0 and "[result] (pass)" in text:
        return "ok"
    return "runtime_exception"


def _derive_success(stdout: str, stderr: str, return_code: int) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    if "[result] (pass)" in text:
        return True
    if "[result] (fail)" in text:
        return False
    return return_code == 0 and "no task left to run" not in text


def _extract_score(stdout: str, success: bool) -> float:
    text = stdout.lower()
    if success:
        return 1.0
    if "[result] (fail)" in text:
        return 0.0
    return 0.0


def _load_task_metadata(config_file: Path) -> dict[str, Any]:
    data = read_json(config_file)
    intent = str(data.get("intent", ""))
    start_url = str(data.get("start_url", ""))
    task_id = str(data.get("task_id", config_file.stem))
    return {"task_id": task_id, "intent": intent, "start_url": start_url}


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
    return count


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
    if task_out_dir.exists():
        shutil.rmtree(task_out_dir)
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
    for key, value in DEFAULT_SITE_ENV.items():
        env.setdefault(key, value)
    env.setdefault("WEBARENA_SKIP_AUTO_LOGIN", "1")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{webarena_root}:{existing_pythonpath}" if existing_pythonpath else str(webarena_root)
    proc = subprocess.run(command, cwd=webarena_root, capture_output=True, text=True, env=env)
    error_code = _classify_error(proc.stdout, proc.stderr, proc.returncode)
    success = _derive_success(proc.stdout, proc.stderr, proc.returncode)
    agent_steps_file = task_out_dir / "agent_steps.jsonl"
    step_count = _count_jsonl_rows(agent_steps_file)
    record = WebArenaRunRecord(
        run_id=run_id,
        task_id=meta["task_id"],
        config_file=str(config_file),
        task_intent=meta["intent"],
        start_url=meta["start_url"],
        model_name=model_name,
        success=success,
        score=_extract_score(proc.stdout, success),
        steps=step_count,
        terminated=True,
        error="" if success else (proc.stderr[-3000:] or proc.stdout[-3000:]),
        artifacts={
            "result_dir": str(task_out_dir),
            "stdout_file": str(task_out_dir / "stdout.log"),
            "stderr_file": str(task_out_dir / "stderr.log"),
            "agent_steps_file": str(agent_steps_file),
        },
        raw={"returncode": proc.returncode, "error_code": error_code},
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
            record = WebArenaRunRecord(
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
            records.append(record)
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

    payload = {
        "run_id": run_id,
        "provider": provider,
        "model_name": model_name,
        "model_endpoint": model_endpoint,
        "config_dir": str(config_dir),
        "webarena_root": str(webarena_root),
        "python_executable": python_executable,
        "task_count": len(task_ids),
        "records": [item.to_dict() for item in records],
    }
    write_json(result_dir / run_id / "batch_summary.json", payload)
    write_jsonl(result_dir / run_id / "batch_records.jsonl", (item.to_dict() for item in records))
    return records


def load_batch_records(batch_dir: Path) -> list[WebArenaRunRecord]:
    path = batch_dir / "batch_summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [WebArenaRunRecord(**item) for item in payload.get("records", [])]

