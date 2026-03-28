#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for full training on shopping_split_manifest.json.
# Usage:
#   ./scripts/run_full_training.sh
#   ./scripts/run_full_training.sh --dry-run
#   ./scripts/run_full_training.sh --config ./configs/default.json --split ./data/splits/shopping_split_manifest.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="./configs/default.json"
SPLIT_PATH="./data/splits/shopping_split_manifest.json"
WEBARENA_CONFIG_DIR="../webarena/config_files"
DRY_RUN=0
RESILIENT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --split)
      SPLIT_PATH="$2"
      shift 2
      ;;
    --webarena-config-dir)
      WEBARENA_CONFIG_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-resilient)
      RESILIENT=0
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "Running preflight doctor..."
python3 "./scripts/doctor.py" --config "$CONFIG_PATH" --split-manifest "$SPLIT_PATH" --webarena-config-dir "$WEBARENA_CONFIG_DIR"

echo "Computing planned bake windows..."
CONFIG_FOR_COUNT="$CONFIG_PATH" SPLIT_FOR_COUNT="$SPLIT_PATH" python3 - <<'PY'
import json, math
import os
from pathlib import Path
root = Path(".")
cfg_path = Path(os.environ["CONFIG_FOR_COUNT"])
split_path = Path(os.environ["SPLIT_FOR_COUNT"])
cfg = json.loads((root / cfg_path).read_text(encoding="utf-8"))
split = json.loads((root / split_path).read_text(encoding="utf-8"))
batch_size = int(cfg.get("batch_size", 20))
train_count = int(split.get("train_count", len(split.get("train_ids", []))))
windows = math.ceil(train_count / max(batch_size, 1))
print(f"train_count={train_count}, batch_size={batch_size}, planned_bakes={windows}")
PY

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Starting DRY-RUN training loop..."
  TRAIN_CMD=(python3 "./scripts/train_loop.py")
  if [[ "$RESILIENT" -eq 1 ]]; then
    TRAIN_CMD=(python3 "./scripts/train_loop_resilient.py")
  fi
  "${TRAIN_CMD[@]}" \
    --config "$CONFIG_PATH" \
    --split-manifest "$SPLIT_PATH" \
    --webarena-config-dir "$WEBARENA_CONFIG_DIR" \
    --bake-backend tinker
else
  echo "Starting LIVE training loop..."
  TRAIN_CMD=(python3 "./scripts/train_loop.py")
  if [[ "$RESILIENT" -eq 1 ]]; then
    TRAIN_CMD=(python3 "./scripts/train_loop_resilient.py")
  fi
  "${TRAIN_CMD[@]}" \
    --config "$CONFIG_PATH" \
    --split-manifest "$SPLIT_PATH" \
    --webarena-config-dir "$WEBARENA_CONFIG_DIR" \
    --bake-backend tinker \
    --live-bake
fi
