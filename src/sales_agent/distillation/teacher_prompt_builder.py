from __future__ import annotations

from sales_agent.schemas.types import DistilledRule


def build_teacher_prompt(distilled_rules: list[DistilledRule], domain_name: str) -> str:
    lines = [
        f"You are an expert {domain_name} agent.",
        "Follow these persistent operating rules:",
    ]
    for idx, rule in enumerate(distilled_rules, start=1):
        lines.append(f"{idx}. {rule.rule_text}")

    lines.extend(
        [
            "",
            "Prioritize correctness over speed.",
            "If evidence is weak, state uncertainty explicitly.",
            "Stop once success criteria are satisfied.",
        ]
    )
    return "\n".join(lines)

