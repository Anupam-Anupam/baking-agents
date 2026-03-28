#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate WebArena .auth storage state files.")
    parser.add_argument("--config", default="./configs/default.json", help="Pipeline config file.")
    parser.add_argument("--split-manifest", default="./data/splits/shopping_split_manifest.json", help="Split manifest path.")
    parser.add_argument("--webarena-config-dir", required=True, help="Directory with WebArena config json files.")
    parser.add_argument("--task-limit", type=int, default=20, help="How many train tasks to validate auth files for.")
    parser.add_argument("--refresh", action="store_true", help="Always re-run auto_login even if auth files exist.")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parents[1]
    src_dir = workspace_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from webarena_bake.runners.preflight import REQUIRED_WEBARENA_ENV_VARS, collect_required_auth_files
    from webarena_bake.runners.webarena_batch_runner import DEFAULT_SITE_ENV
    from webarena_bake.utils.io import read_json

    cfg = read_json((workspace_dir / args.config).resolve())
    split = read_json((workspace_dir / args.split_manifest).resolve())
    task_ids = list(split.get("train_ids", []))[: max(args.task_limit, 1)]
    webarena_root = (workspace_dir / cfg["webarena_root"]).resolve()
    webarena_config_dir = Path(args.webarena_config_dir).resolve()
    python_executable = str((workspace_dir / cfg.get("webarena_python_executable", "python")).resolve())

    auth_dir = webarena_root / ".auth"
    auth_dir.mkdir(parents=True, exist_ok=True)

    required_auth = collect_required_auth_files(webarena_root, webarena_config_dir, task_ids)
    missing_before = [str(path) for path in required_auth if not path.exists()]

    generated = False
    if args.refresh or missing_before:
        env = os.environ.copy()
        for key, value in DEFAULT_SITE_ENV.items():
            env.setdefault(key, value)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{webarena_root}:{existing_pythonpath}" if existing_pythonpath else str(webarena_root)
        missing_env = [name for name in REQUIRED_WEBARENA_ENV_VARS if not env.get(name)]
        if missing_env:
            raise RuntimeError(f"Missing required environment vars for auto_login: {', '.join(missing_env)}")

        cmd = [python_executable, "browser_env/auto_login.py", "--auth_folder", str(auth_dir), "--site_list", "all"]
        proc = subprocess.run(cmd, cwd=webarena_root, env=env, capture_output=True, text=True, check=False)
        generated = proc.returncode == 0
        if proc.returncode != 0:
            raise RuntimeError(
                "auto_login.py failed.\n"
                f"stdout:\n{proc.stdout[-3000:]}\n\nstderr:\n{proc.stderr[-3000:]}"
            )

    missing_after = [str(path) for path in required_auth if not path.exists()]
    payload = {
        "generated": generated,
        "task_count_checked": len(task_ids),
        "required_auth_files": [str(path) for path in required_auth],
        "missing_before": missing_before,
        "missing_after": missing_after,
    }
    print(json.dumps(payload, indent=2))
    if missing_after:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
