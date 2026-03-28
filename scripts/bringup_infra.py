#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import requests

SERVICES = [
    ("shopping", "http://localhost:7770"),
    ("shopping_admin", "http://localhost:7780/admin"),
    ("forum", "http://localhost:9999"),
    ("gitlab", "http://localhost:8023"),
]


def _docker_ready() -> bool:
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
        return proc.returncode == 0
    except FileNotFoundError:
        return False


def _probe(url: str, timeout: float = 2.0) -> dict:
    try:
        response = requests.get(url, timeout=timeout)
        return {"ok": response.status_code < 500, "status": response.status_code, "error": ""}
    except Exception as exc:  # pragma: no cover - diagnostics path
        return {"ok": False, "status": 0, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-effort WebArena docker bring-up and health checks.")
    parser.add_argument("--retries", type=int, default=6, help="Number of health-check retry rounds.")
    parser.add_argument("--sleep-seconds", type=int, default=5, help="Delay between retry rounds.")
    parser.add_argument("--output", default="./results/state/infra_report.json", help="Where to write report.")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parents[1]
    report: dict = {
        "docker_ready": _docker_ready(),
        "started_containers": [],
        "health": [],
    }
    if not report["docker_ready"]:
        output_path = (workspace_dir / args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

    for name, _ in SERVICES:
        proc = subprocess.run(["docker", "start", name], capture_output=True, text=True, check=False)
        report["started_containers"].append(
            {"name": name, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
        )

    for _ in range(max(args.retries, 1)):
        health = []
        for name, url in SERVICES:
            status = _probe(url)
            health.append({"name": name, "url": url, **status})
        report["health"] = health
        if all(item["ok"] for item in health):
            break
        time.sleep(max(args.sleep_seconds, 1))

    output_path = (workspace_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(item["ok"] for item in report["health"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
