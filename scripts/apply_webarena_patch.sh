#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_PATH="$ROOT_DIR/patches/webarena_required.patch"
WEBARENA_DIR="${1:-$ROOT_DIR/../webarena}"

if [[ ! -f "$PATCH_PATH" ]]; then
  echo "Missing patch file: $PATCH_PATH" >&2
  exit 1
fi

if [[ ! -d "$WEBARENA_DIR/.git" ]]; then
  echo "WebArena repo not found at: $WEBARENA_DIR" >&2
  echo "Pass path explicitly: ./scripts/apply_webarena_patch.sh /path/to/webarena" >&2
  exit 1
fi

echo "Applying patch to: $WEBARENA_DIR"
git -C "$WEBARENA_DIR" apply --check "$PATCH_PATH"
git -C "$WEBARENA_DIR" apply "$PATCH_PATH"
echo "Patch applied successfully."
echo "Changed files in webarena:"
git -C "$WEBARENA_DIR" status --short
