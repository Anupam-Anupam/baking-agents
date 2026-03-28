from __future__ import annotations

from webarena_bake.observer.observer_model import call_observer_endpoint, heuristic_observe
from webarena_bake.schemas.types import Lesson, WebArenaRunRecord


def extract_lessons(
    records: list[WebArenaRunRecord],
    task_family: str = "shopping",
    observer_endpoint: str | None = None,
    observer_model: str | None = None,
    observer_api_key: str | None = None,
) -> list[Lesson]:
    lessons: list[Lesson] = []
    for record in records:
        decision = (
            call_observer_endpoint(record, observer_endpoint, observer_model, observer_api_key)
            if observer_endpoint and observer_model
            else heuristic_observe(record)
        )
        status = "success" if record.success else "fail"

        lesson_id = f"lesson_{record.run_id}_{record.task_id}"
        lessons.append(
            Lesson(
                lesson_id=lesson_id,
                run_id=record.run_id,
                task_id=record.task_id,
                task_family=task_family,
                status=status,
                failure_type=decision.failure_type,
                rule_text=decision.rule_text,
                confidence=decision.confidence,
                supporting_runs=[record.run_id],
            )
        )
    return lessons

