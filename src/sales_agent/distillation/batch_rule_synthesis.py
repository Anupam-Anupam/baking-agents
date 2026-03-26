from __future__ import annotations

from collections import Counter

from sales_agent.distillation.cluster_lessons import cluster_lessons
from sales_agent.schemas.types import DistilledRule, Lesson


def _resolve_task_family(cluster: list[Lesson]) -> str:
    counts = Counter(lesson.task_family for lesson in cluster)
    return counts.most_common(1)[0][0]


def synthesize_distilled_rules(lessons: list[Lesson], min_support: int = 2) -> list[DistilledRule]:
    clusters = cluster_lessons(lessons)
    distilled: list[DistilledRule] = []
    for rule_text, grouped in clusters.items():
        if len(grouped) < min_support:
            continue
        avg_conf = sum(lesson.confidence for lesson in grouped) / len(grouped)
        distilled.append(
            DistilledRule(
                rule_id=f"rule_{len(distilled) + 1:04d}",
                task_family=_resolve_task_family(grouped),
                rule_text=grouped[0].rule_text,
                support_count=len(grouped),
                avg_confidence=avg_conf,
                source_lesson_ids=[lesson.lesson_id for lesson in grouped],
            )
        )

    distilled.sort(key=lambda r: (r.support_count, r.avg_confidence), reverse=True)
    return distilled

