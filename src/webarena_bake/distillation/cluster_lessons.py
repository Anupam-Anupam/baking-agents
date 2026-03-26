from __future__ import annotations

import re
from collections import defaultdict

from webarena_bake.schemas.types import Lesson


def _normalize_rule(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def cluster_lessons(lessons: list[Lesson]) -> dict[str, list[Lesson]]:
    clusters: dict[str, list[Lesson]] = defaultdict(list)
    for lesson in lessons:
        clusters[_normalize_rule(lesson.rule_text)].append(lesson)
    return dict(clusters)

