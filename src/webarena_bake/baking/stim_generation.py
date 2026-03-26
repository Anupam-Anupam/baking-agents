from __future__ import annotations

from pathlib import Path

from webarena_bake.utils.io import write_jsonl


SHOPPING_TEMPLATES = [
    "Find the requested product and verify price and availability.",
    "Complete a shopping support workflow with minimal navigation steps.",
    "Resolve the user's shopping task while collecting only required evidence.",
]


def generate_stim_jsonl(output_path: Path, count: int = 200) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i in range(count):
        rows.append(
            {
                "id": f"shopping_stim_{i:04d}",
                "prompt": SHOPPING_TEMPLATES[i % len(SHOPPING_TEMPLATES)],
                "task_family": "shopping",
            }
        )
    write_jsonl(output_path, rows)
    return rows

