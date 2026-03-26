from __future__ import annotations

from pathlib import Path

from webarena_bake.distillation.batch_rule_synthesis import synthesize_rules
from webarena_bake.distillation.prompt_recipe_builder import build_prompt_recipe
from webarena_bake.schemas.types import Lesson, PromptRecipe
from webarena_bake.utils.io import write_json, write_jsonl


def distill_window(lessons: list[Lesson], output_dir: Path, recipe_version: str) -> PromptRecipe:
    rules = synthesize_rules(lessons=lessons, min_support=2, min_confidence=0.65, max_contradictions=0)
    if not rules and lessons:
        rules = synthesize_rules(lessons=lessons, min_support=1, min_confidence=0.5, max_contradictions=0)

    recipe = build_prompt_recipe(
        recipe_version=recipe_version,
        distilled_rules=rules,
        teacher_prompt_name=f"shopping_teacher_{recipe_version}",
        student_prompt_name=f"shopping_student_{recipe_version}",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "atomic_lessons.jsonl", (item.to_dict() for item in lessons))
    write_json(output_dir / "distilled_rules.json", {"rules": [rule.to_dict() for rule in rules]})
    write_json(output_dir / "bread_recipe.json", recipe.to_dict())
    (output_dir / "teacher_prompt.txt").write_text(recipe.teacher_prompt_text, encoding="utf-8")
    (output_dir / "student_prompt.txt").write_text(recipe.student_prompt_text, encoding="utf-8")
    return recipe

