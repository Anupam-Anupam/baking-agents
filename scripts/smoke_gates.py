#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, dict]:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    payload = {
        "command": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2500:],
        "stderr_tail": proc.stderr[-2500:],
    }
    return proc.returncode, payload


def _safe_smoke_task_ids(config_dir: Path, candidate_ids: list[str]) -> list[str]:
    safe: list[str] = []
    for task_id in candidate_ids:
        cfg_path = config_dir / f"{task_id}.json"
        if not cfg_path.exists():
            continue
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        eval_types = list(payload.get("eval", {}).get("eval_types", []))
        # Prefer deterministic string-match tasks that do not need extra LLM judge calls.
        if eval_types == ["string_match"]:
            safe.append(task_id)
    return safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged WebArena smoke-test gates.")
    parser.add_argument("--config", default="./configs/default.json", help="Pipeline config file.")
    parser.add_argument("--split-manifest", default="./data/splits/shopping_split_manifest.json", help="Split manifest path.")
    parser.add_argument("--webarena-config-dir", required=True, help="Directory with WebArena config json files.")
    parser.add_argument("--output", default="./results/state/smoke_gates.json", help="Output JSON report path.")
    parser.add_argument("--provider", default=None, help="Override provider.")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parents[1]
    src_dir = workspace_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from webarena_bake.runners.webarena_batch_runner import DEFAULT_SITE_ENV, run_batch
    from webarena_bake.runners.train_loop import LoopConfig, run_train_loop
    from webarena_bake.utils.io import read_json, write_json

    cfg = read_json((workspace_dir / args.config).resolve())
    split = read_json((workspace_dir / args.split_manifest).resolve())
    provider = args.provider or cfg.get("provider", "openai_compatible")
    train_ids = list(split.get("train_ids", []))
    if not train_ids:
        raise RuntimeError("No train_ids found in split manifest.")
    safe_ids = _safe_smoke_task_ids(Path(args.webarena_config_dir).resolve(), train_ids)
    if not safe_ids:
        raise RuntimeError("Could not find any safe string_match tasks for smoke tests.")

    webarena_root = (workspace_dir / cfg["webarena_root"]).resolve()
    config_dir = Path(args.webarena_config_dir).resolve()
    python_exec = str((workspace_dir / cfg.get("webarena_python_executable", "python")).resolve())

    env = os.environ.copy()
    for key, value in DEFAULT_SITE_ENV.items():
        env.setdefault(key, value)
    env.setdefault("WEBARENA_SKIP_AUTO_LOGIN", "1")

    report: dict[str, dict] = {}
    infra_blocking_error_codes = {"infra_service_unreachable", "webarena_no_task_selected"}

    # Gate 1: raw WebArena run.py one task
    gate1_task = safe_ids[0]
    smoke_root = (workspace_dir / "results" / "smoke").resolve()
    smoke_root.mkdir(parents=True, exist_ok=True)
    gate1_temp_dir = Path(tempfile.mkdtemp(prefix="gate1_", dir=str(smoke_root)))
    gate1_cmd = [
        python_exec,
        "run.py",
        "--instruction_path",
        cfg["webarena_instruction_path"],
        "--test_start_idx",
        gate1_task,
        "--test_end_idx",
        str(int(gate1_task) + 1),
        "--provider",
        provider,
        "--model",
        cfg["policy_model_name"],
        "--model_endpoint",
        cfg["policy_model_endpoint"],
        "--result_dir",
        str(gate1_temp_dir.resolve()),
    ]
    returncode, payload = _run(gate1_cmd, cwd=webarena_root, env=env)
    combined = f"{payload['stdout_tail']}\n{payload['stderr_tail']}".lower()
    executed_task = "[result] (pass)" in combined or "[result] (fail)" in combined
    gate1_ok = returncode == 0 and executed_task and "no task left to run" not in combined
    report["gate1_raw_webarena"] = {"ok": gate1_ok, **payload}
    if not gate1_ok:
        output = (workspace_dir / args.output).resolve()
        write_json(output, report)
        raise SystemExit(1)

    # Gate 2: run_batch with 1 task
    gate2_records = run_batch(
        webarena_root=webarena_root,
        config_dir=config_dir,
        task_ids=[safe_ids[0]],
        run_id="smoke_gate2",
        instruction_path=cfg["webarena_instruction_path"],
        provider=provider,
        model_name=cfg["policy_model_name"],
        model_endpoint=cfg["policy_model_endpoint"],
        result_dir=workspace_dir / "results" / "smoke" / "runs",
        python_executable=python_exec,
    )
    gate2_ok = bool(
        gate2_records
        and gate2_records[0].raw.get("returncode", 1) == 0
        and gate2_records[0].raw.get("error_code", "") not in infra_blocking_error_codes
    )
    report["gate2_batch_single"] = {
        "ok": gate2_ok,
        "task_id": safe_ids[0],
        "records": [item.to_dict() for item in gate2_records],
    }
    if not gate2_ok:
        output = (workspace_dir / args.output).resolve()
        write_json(output, report)
        raise SystemExit(1)

    # Gate 3: run_batch with a stable small subset
    gate3_tasks = safe_ids[: min(2, len(safe_ids))]
    gate3_records = run_batch(
        webarena_root=webarena_root,
        config_dir=config_dir,
        task_ids=gate3_tasks,
        run_id="smoke_gate3",
        instruction_path=cfg["webarena_instruction_path"],
        provider=provider,
        model_name=cfg["policy_model_name"],
        model_endpoint=cfg["policy_model_endpoint"],
        result_dir=workspace_dir / "results" / "smoke" / "runs",
        python_executable=python_exec,
    )
    gate3_ok = all(
        item.raw.get("returncode", 1) == 0
        and item.raw.get("error_code", "") not in infra_blocking_error_codes
        for item in gate3_records
    )
    report["gate3_batch_small"] = {
        "ok": gate3_ok,
        "task_ids": gate3_tasks,
        "records": [item.to_dict() for item in gate3_records],
    }
    if not gate3_ok:
        output = (workspace_dir / args.output).resolve()
        write_json(output, report)
        raise SystemExit(1)

    # Gate 4: one train-loop window on the same stable subset
    gate4_tasks = safe_ids[: min(2, len(safe_ids))]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_split = Path(tmp_dir) / "smoke_split.json"
        write_json(
            tmp_split,
            {
                "seed": split.get("seed", 42),
                "train_ratio": split.get("train_ratio", 0.8),
                "total": len(gate4_tasks),
                "train_count": len(gate4_tasks),
                "val_count": 0,
                "train_ids": gate4_tasks,
                "val_ids": [],
            },
        )
        result = run_train_loop(
            workspace_dir=workspace_dir,
            config=LoopConfig(
                webarena_root=webarena_root,
                webarena_config_dir=config_dir,
                split_manifest_path=tmp_split,
                instruction_path=cfg["webarena_instruction_path"],
                provider=provider,
                initial_model_name=cfg["policy_model_name"],
                model_endpoint=cfg["policy_model_endpoint"],
                batch_size=max(20, len(gate4_tasks)),
                bread_repo_name=cfg["bread_repo_name"],
                bread_base_model=cfg.get("bread_base_model"),
                dry_run_bake=bool(cfg.get("dry_run_bake", True)),
                observer_endpoint=cfg.get("observer_model_endpoint"),
                observer_model=cfg.get("observer_model_name"),
                observer_api_key=cfg.get("observer_api_key"),
                webarena_python_executable=python_exec,
                run_preflight_checks=bool(cfg.get("run_preflight_checks", True)),
            ),
        )
    report["gate4_one_window"] = {"ok": True, "result": result}

    output = (workspace_dir / args.output).resolve()
    write_json(output, report)
    print(json.dumps({"ok": True, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
