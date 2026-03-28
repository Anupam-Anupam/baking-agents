#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WebArena bake preflight diagnostics.")
    parser.add_argument("--config", default="./configs/default.json", help="Pipeline config file.")
    parser.add_argument("--split-manifest", default="./data/splits/shopping_split_manifest.json", help="Split manifest path.")
    parser.add_argument("--webarena-config-dir", required=True, help="Directory with WebArena config json files.")
    parser.add_argument("--provider", default=None, help="Override provider.")
    parser.add_argument("--task-limit", type=int, default=20, help="Number of train tasks used for preflight auth/service checks.")
    parser.add_argument("--output", default="./results/state/doctor_report.json", help="Where to write diagnostics.")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parents[1]
    src_dir = workspace_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from webarena_bake.runners.preflight import run_preflight
    from webarena_bake.utils.io import read_json, write_json

    cfg = read_json((workspace_dir / args.config).resolve())
    split = read_json((workspace_dir / args.split_manifest).resolve())
    task_ids = list(split.get("train_ids", []))[: max(args.task_limit, 1)]
    provider = args.provider or cfg.get("provider", "openai_compatible")

    report = run_preflight(
        webarena_root=(workspace_dir / cfg["webarena_root"]).resolve(),
        webarena_config_dir=Path(args.webarena_config_dir).resolve(),
        task_ids=task_ids,
        policy_model_endpoint=cfg["policy_model_endpoint"],
        observer_model_endpoint=cfg.get("observer_model_endpoint"),
        provider=provider,
    )
    output_path = (workspace_dir / args.output).resolve()
    write_json(output_path, report)
    print(json.dumps({"ok": report["ok"], "output": str(output_path), "hints": report.get("hints", [])}, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
