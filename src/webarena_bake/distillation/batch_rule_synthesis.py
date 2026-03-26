from __future__ import annotations

from collections import Counter

from webarena_bake.distillation.cluster_lessons import cluster_lessons
from webarena_bake.distillation.contradictions import contradiction_score
from webarena_bake.schemas.types import DistilledRule, Lesson


def _dominant_family(group: list[Lesson]) -> str:
    counts = Counter(item.task_family for item in group)
    return counts.most_common(1)[0][0]


def synthesize_rules(
    lessons: list[Lesson],
    min_support: int = 2,
    min_confidence: float = 0.65,
    max_contradictions: int = 0,
) -> list[DistilledRule]:
    clusters = cluster_lessons(lessons)
    rules: list[DistilledRule] = []
    for _, group in clusters.items():
        if len(group) < min_support:
            continue
        avg_conf = sum(item.confidence for item in group) / len(group)
        if avg_conf < min_confidence:
            continue
        contradictions = contradiction_score(group)
        if contradictions > max_contradictions:
            continue
        rules.append(
            DistilledRule(
                rule_id=f"rule_{len(rules)+1:04d}",
                task_family=_dominant_family(group),
                rule_text=group[0].rule_text,
                support_count=len(group),
                avg_confidence=avg_conf,
                contradiction_count=contradictions,
                source_lesson_ids=[item.lesson_id for item in group],
            )
        )
    rules.sort(key=lambda item: (item.support_count, item.avg_confidence), reverse=True)
    return rules

