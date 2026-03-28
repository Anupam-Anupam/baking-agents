#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation ablation matrix.")
    parser.add_argument("--config", default="./configs/default.json", help="Pipeline config file.")
    parser.add_argument("--split-manifest", default="./data/splits/shopping_split_manifest.json", help="Split manifest path.")
    parser.add_argument("--webarena-config-dir", required=True, help="Directory with WebArena config json files.")
    parser.add_argument("--baked-model-name", required=True, help="Baked model name/path to evaluate.")
    parser.add_argument("--provider", default=None, help="Model provider name for WebArena.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional cap for validation tasks.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip startup preflight checks.")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parents[1]
    src_dir = workspace_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from webarena_bake.evaluation.ablation_runner import AblationConfig, run_ablation
    from webarena_bake.utils.io import read_json

    cfg = read_json((workspace_dir / args.config).resolve())
    provider = args.provider or cfg.get("provider", "openai_compatible")
    result = run_ablation(
        workspace_dir=workspace_dir,
        config=AblationConfig(
            webarena_root=(workspace_dir / cfg["webarena_root"]).resolve(),
            webarena_config_dir=Path(args.webarena_config_dir).resolve(),
            split_manifest_path=(workspace_dir / args.split_manifest).resolve(),
            instruction_path=cfg["webarena_instruction_path"],
            provider=provider,
            base_model_name=cfg["policy_model_name"],
            baked_model_name=args.baked_model_name,
            model_endpoint=cfg["policy_model_endpoint"],
            max_tasks=args.max_tasks,
            webarena_python_executable=cfg.get("webarena_python_executable", "python"),
            observer_endpoint=cfg.get("observer_model_endpoint"),
            run_preflight_checks=False if args.skip_preflight else bool(cfg.get("run_preflight_checks", True)),
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

