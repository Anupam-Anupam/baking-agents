#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative 20-task observer-bake training loop.")
    parser.add_argument("--config", default="./configs/default.json", help="Pipeline config file.")
    parser.add_argument("--split-manifest", default="./data/splits/shopping_split_manifest.json", help="Split manifest path.")
    parser.add_argument("--webarena-config-dir", required=True, help="Directory with WebArena config json files.")
    parser.add_argument("--provider", default=None, help="Model provider name for WebArena run.py")
    parser.add_argument("--live-bake", action="store_true", help="Enable live bake calls (instead of dry-run prep).")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip startup preflight checks.")
    parser.add_argument("--no-resume-from-state", action="store_true", help="Disable automatic resume from existing lineage.json state.")
    parser.add_argument("--bake-backend", default=None, choices=["bread_sdk", "tinker"], help="Bake backend implementation.")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parents[1]
    # Support keys/config in either project root or webarena-bake root.
    for env_path in (workspace_dir / ".env", workspace_dir.parent / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)
    src_dir = workspace_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from webarena_bake.runners.train_loop import LoopConfig, run_train_loop
    from webarena_bake.utils.io import read_json

    cfg = read_json((workspace_dir / args.config).resolve())
    provider = args.provider or cfg.get("provider", "openai_compatible")
    bake_backend = args.bake_backend or cfg.get("bake_backend", "bread_sdk")
    loop_cfg = LoopConfig(
        webarena_root=(workspace_dir / cfg["webarena_root"]).resolve(),
        webarena_config_dir=Path(args.webarena_config_dir).resolve(),
        split_manifest_path=(workspace_dir / args.split_manifest).resolve(),
        instruction_path=cfg["webarena_instruction_path"],
        provider=provider,
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
        run_preflight_checks=False if args.skip_preflight else bool(cfg.get("run_preflight_checks", True)),
        bake_backend=bake_backend,
        tinker_repo_path=cfg.get("tinker_repo_path"),
        tinker_python_executable=cfg.get("tinker_python_executable", "python3"),
        tinker_num_epochs=int(cfg.get("tinker_num_epochs", 1)),
        tinker_top_k=int(cfg.get("tinker_top_k", 20)),
        wandb_project=cfg.get("wandb_project", "webarena-bake"),
        wandb_entity=cfg.get("wandb_entity"),
        wandb_single_run=bool(cfg.get("wandb_single_run", True)),
        wandb_run_name=cfg.get("wandb_run_name"),
        wandb_enable_child_runs=bool(cfg.get("wandb_enable_child_runs", False)),
        resume_from_state=not args.no_resume_from_state,
    )
    result = run_train_loop(workspace_dir, loop_cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

