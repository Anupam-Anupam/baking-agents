from __future__ import annotations

from sales_agent.distillation.student_prompt_builder import build_student_prompt
from sales_agent.distillation.teacher_prompt_builder import build_teacher_prompt
from sales_agent.schemas.types import DistilledRule, PromptRecipe


def build_prompt_recipe(
    recipe_version: str,
    distilled_rules: list[DistilledRule],
    task_families: list[str],
    teacher_prompt_name: str,
    student_prompt_name: str,
    domain_name: str,
) -> PromptRecipe:
    teacher_prompt_text = build_teacher_prompt(distilled_rules, domain_name=domain_name)
    student_prompt_text = build_student_prompt(always_on=True)
    return PromptRecipe(
        recipe_version=recipe_version,
        teacher_prompt_name=teacher_prompt_name,
        student_prompt_name=student_prompt_name,
        teacher_prompt_text=teacher_prompt_text,
        student_prompt_text=student_prompt_text,
        rule_ids=[rule.rule_id for rule in distilled_rules],
        task_families=sorted(set(task_families)),
    )

