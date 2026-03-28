#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train_loop.py with automatic crash retries and stateful resume.")
    parser.add_argument("--config", default="./configs/default.json")
    parser.add_argument("--split-manifest", default="./data/splits/shopping_split_manifest.json")
    parser.add_argument("--webarena-config-dir", required=True)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--bake-backend", default="tinker", choices=["bread_sdk", "tinker"])
    parser.add_argument("--live-bake", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("--restart-delay-sec", type=int, default=10)
    parser.add_argument("--no-resume-from-state", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_loop_script = root / "scripts" / "train_loop.py"

    base_cmd = [
        sys.executable,
        str(train_loop_script),
        "--config",
        args.config,
        "--split-manifest",
        args.split_manifest,
        "--webarena-config-dir",
        args.webarena_config_dir,
        "--bake-backend",
        args.bake_backend,
    ]
    if args.provider:
        base_cmd.extend(["--provider", args.provider])
    if args.live_bake:
        base_cmd.append("--live-bake")
    if args.skip_preflight:
        base_cmd.append("--skip-preflight")
    if args.no_resume_from_state:
        base_cmd.append("--no-resume-from-state")

    for attempt in range(1, args.max_restarts + 2):
        print(f"[resilient] launch attempt={attempt}", flush=True)
        proc = subprocess.run(base_cmd, cwd=root, check=False)
        if proc.returncode == 0:
            print("[resilient] completed successfully", flush=True)
            return
        if attempt > args.max_restarts:
            raise SystemExit(f"[resilient] giving up after {attempt - 1} restarts (last rc={proc.returncode})")
        print(
            f"[resilient] train_loop exited rc={proc.returncode}, restarting in {args.restart_delay_sec}s...",
            flush=True,
        )
        time.sleep(max(args.restart_delay_sec, 0))


if __name__ == "__main__":
    main()
