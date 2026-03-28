#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _extract_key_lines(text: str) -> list[str]:
    keep = ("[Intent]", "[Result]", "[Unhandled Error]", "[OpenAI Error]", "Average score")
    lines: list[str] = []
    for line in text.splitlines():
        if any(token in line for token in keep):
            lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Show compact teacher/run logs for a window.")
    parser.add_argument("--window", required=True, help="Window id, e.g. window_001")
    parser.add_argument("--root", default="./results/runs", help="Runs root directory")
    parser.add_argument("--limit", type=int, default=10, help="Max tasks to print")
    args = parser.parse_args()

    window_dir = (Path(args.root) / args.window).resolve()
    summary_path = window_dir / "batch_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records = list(payload.get("records", []))
    print(f"window={args.window} tasks={len(records)}")

    shown = 0
    for rec in records:
        if shown >= max(args.limit, 1):
            break
        task_id = str(rec.get("task_id", ""))
        success = bool(rec.get("success", False))
        intent = str(rec.get("task_intent", "")).strip()
        raw = dict(rec.get("raw", {}))
        error_code = str(raw.get("error_code", ""))
        stdout_path = window_dir / task_id / "stdout.log"
        stderr_path = window_dir / task_id / "stderr.log"
        stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        key_lines = _extract_key_lines(stdout_text + "\n" + stderr_text)

        print(f"\n--- task={task_id} success={success} error_code={error_code}")
        print(f"intent={intent}")
        for line in key_lines[:8]:
            print(line)
        shown += 1


if __name__ == "__main__":
    main()
