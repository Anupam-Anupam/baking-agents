from __future__ import annotations

from webarena_bake.schemas.types import DistilledRule


def build_teacher_prompt(rules: list[DistilledRule], domain_name: str = "shopping web navigation") -> str:
    lines = [
        f"You are an expert autonomous agent for {domain_name}.",
        "Apply these persistent operating rules:",
    ]
    for idx, rule in enumerate(rules, start=1):
        lines.append(f"{idx}. {rule.rule_text}")
    lines.extend(
        [
            "",
            "Always prefer verified page evidence before final answers.",
            "If objective is complete, stop immediately.",
            "If confidence is low, explicitly state uncertainty.",
        ]
    )
    return "\n".join(lines)

