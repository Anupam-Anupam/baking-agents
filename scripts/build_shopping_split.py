#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shopping-only 80/20 split from WebArena config files.")
    parser.add_argument("--config-dir", required=True, help="Directory containing WebArena task JSON files.")
    parser.add_argument("--output-dir", default="./data/splits", help="Output directory for manifests.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stable split.")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parents[1]
    src_dir = workspace_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from webarena_bake.runners.splits import build_shopping_split

    manifest = build_shopping_split(
        config_dir=Path(args.config_dir).resolve(),
        output_dir=(workspace_dir / args.output_dir).resolve(),
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

