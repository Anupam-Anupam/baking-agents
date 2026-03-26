from __future__ import annotations

from typing import Any


def build_rollout_generators(stim_rows: list[dict[str, str]], model: str = "claude-sonnet-4-5-20250929") -> list[dict[str, Any]]:
    """
    Build a simple hardcoded generator block from stim prompts.
    """
    questions = [row["prompt"] for row in stim_rows]
    return [
        {
            "type": "hardcoded",
            "numq": len(questions),
            "questions": questions,
            "model": model,
            "temperature": 0.4,
        }
    ]

