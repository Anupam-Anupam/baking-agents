#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative observer-bake training loop.")
    parser.add_argument("--config", default="./configs/default.json", help="Pipeline config file.")
    parser.add_argument("--split-manifest", default="./data/splits/shopping_split_manifest.json", help="Split manifest path.")
    parser.add_argument("--webarena-config-dir", required=True, help="Directory with WebArena config json files.")
    parser.add_argument("--provider", default="openai", help="Model provider name for WebArena run.py")
    parser.add_argument("--live-bake", action="store_true", help="Enable live Bread bake calls.")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parents[1]
    src_dir = workspace_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from webarena_bake.runners.train_loop import LoopConfig, run_train_loop
    from webarena_bake.utils.io import read_json

    cfg = read_json((workspace_dir / args.config).resolve())
    result = run_train_loop(
        workspace_dir,
        LoopConfig(
            webarena_root=(workspace_dir / cfg["webarena_root"]).resolve(),
            webarena_config_dir=Path(args.webarena_config_dir).resolve(),
            split_manifest_path=(workspace_dir / args.split_manifest).resolve(),
            instruction_path=cfg["webarena_instruction_path"],
            provider=args.provider,
            initial_model_name=cfg["policy_model_name"],
            model_endpoint=cfg["policy_model_endpoint"],
            batch_size=int(cfg["batch_size"]),
            bread_repo_name=cfg["bread_repo_name"],
            bread_base_model=cfg.get("bread_base_model"),
            dry_run_bake=False if args.live_bake else bool(cfg.get("dry_run_bake", True)),
            observer_endpoint=cfg.get("observer_model_endpoint"),
            observer_model=cfg.get("observer_model_name"),
            observer_api_key=cfg.get("observer_api_key"),
            webarena_python_executable=cfg.get("webarena_python_executable", "python"),
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

