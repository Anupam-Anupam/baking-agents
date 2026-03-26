from __future__ import annotations

from pathlib import Path

from sales_agent.utils.io import write_jsonl


DEFAULT_TEMPLATES: dict[str, list[str]] = {
    "qubit_control": [
        "Given a random qubit state, choose gates to reach |0> quickly and safely.",
        "Improve fidelity to the target state while minimizing unnecessary actions.",
        "Select the next best gate to maximize progress toward the solve threshold.",
    ],
    "lead_discovery": [
        "Find verified decision makers for a mid-market SaaS company.",
        "Collect schema-complete lead data with confidence annotations.",
    ],
}


def generate_stim_jsonl(task_family: str, output_path: Path, count: int = 50) -> list[dict[str, str]]:
    templates = DEFAULT_TEMPLATES.get(task_family, [f"Solve a {task_family} task correctly."])
    rows: list[dict[str, str]] = []
    for idx in range(count):
        rows.append(
            {
                "id": f"{task_family}_stim_{idx:04d}",
                "prompt": templates[idx % len(templates)],
                "task_family": task_family,
            }
        )
    write_jsonl(output_path, rows)
    return rows

