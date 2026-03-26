#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run observer -> distill -> Bread bake pipeline for qubit trajectories.")
    parser.add_argument("--eval-json", required=True, help="Path to evaluation JSON produced by run_evaluation.py")
    parser.add_argument("--run-id", required=True, help="Unique run id, e.g. batch_001")
    parser.add_argument("--repo-name", required=True, help="Bread repo name")
    parser.add_argument("--base-model", default=None, help="Optional Bread base model")
    parser.add_argument("--live", action="store_true", help="Run real Bread API calls (default is dry-run).")
    parser.add_argument("--min-support", type=int, default=2, help="Minimum support for distilled rules.")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parents[1]
    src_dir = workspace_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from sales_agent.runners.qubit_learning_to_bake import run_learning_to_bake

    result = run_learning_to_bake(
        workspace_dir=workspace_dir,
        eval_json_path=Path(args.eval_json).resolve(),
        run_id=args.run_id,
        repo_name=args.repo_name,
        base_model=args.base_model,
        dry_run=not args.live,
        min_support=args.min_support,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

