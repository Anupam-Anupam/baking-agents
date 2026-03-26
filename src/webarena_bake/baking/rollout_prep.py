from __future__ import annotations

from typing import Any


def build_generators(stim_rows: list[dict[str, str]], model: str = "claude-sonnet-4-5-20250929") -> list[dict[str, Any]]:
    return [
        {
            "type": "hardcoded",
            "numq": len(stim_rows),
            "questions": [row["prompt"] for row in stim_rows],
            "model": model,
            "temperature": 0.4,
        }
    ]

