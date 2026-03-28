from __future__ import annotations

from webarena_bake.schemas.types import Lesson


NEGATION_TOKENS = ("avoid", "never", "do not", "don't")


def contradiction_score(group: list[Lesson]) -> int:
    """
    Simple lexical proxy for contradictory guidance.
    """
    has_neg = any(any(token in lesson.rule_text.lower() for token in NEGATION_TOKENS) for lesson in group)
    has_pos = any(not any(token in lesson.rule_text.lower() for token in NEGATION_TOKENS) for lesson in group)
    return 1 if has_neg and has_pos else 0

