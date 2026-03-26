from __future__ import annotations

from webarena_bake.distillation.student_prompt_builder import build_student_prompt
from webarena_bake.distillation.teacher_prompt_builder import build_teacher_prompt
from webarena_bake.schemas.types import DistilledRule, PromptRecipe


def build_prompt_recipe(
    recipe_version: str,
    distilled_rules: list[DistilledRule],
    teacher_prompt_name: str,
    student_prompt_name: str,
) -> PromptRecipe:
    return PromptRecipe(
        recipe_version=recipe_version,
        teacher_prompt_name=teacher_prompt_name,
        student_prompt_name=student_prompt_name,
        teacher_prompt_text=build_teacher_prompt(distilled_rules),
        student_prompt_text=build_student_prompt(always_on=True),
        rule_ids=[rule.rule_id for rule in distilled_rules],
        task_families=sorted({rule.task_family for rule in distilled_rules}) or ["shopping"],
    )

