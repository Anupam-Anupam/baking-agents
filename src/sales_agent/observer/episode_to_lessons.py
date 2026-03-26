from __future__ import annotations

from typing import Iterable

from sales_agent.schemas.types import Lesson, Trajectory


def _fidelity_trend(trajectory: Trajectory) -> float:
    if not trajectory.steps:
        return 0.0
    return trajectory.steps[-1].fidelity - trajectory.steps[0].fidelity


def _dominant_action(trajectory: Trajectory) -> int | None:
    if not trajectory.steps:
        return None
    counts: dict[int, int] = {}
    for step in trajectory.steps:
        counts[step.action] = counts.get(step.action, 0) + 1
    return max(counts, key=counts.get)


def extract_atomic_lessons(trajectories: Iterable[Trajectory]) -> list[Lesson]:
    """
    Convert completed trajectories into machine-readable lessons.

    Heuristics are intentionally simple and deterministic so the output is easy
    to inspect and improve before replacing this with an LLM observer.
    """
    lessons: list[Lesson] = []
    for trajectory in trajectories:
        trend = _fidelity_trend(trajectory)
        dominant_action = _dominant_action(trajectory)
        status = "success" if trajectory.solved else "fail"

        if trajectory.solved:
            rule_text = (
                "When fidelity is improving, continue exploiting recent high-gain gate patterns "
                "and stop once solve threshold is reached."
            )
            failure_type = "none"
            confidence = min(0.95, 0.6 + max(0.0, trend))
        elif trend <= 0.0:
            rule_text = (
                "If fidelity stalls for multiple steps, change gate family instead of repeating "
                "the same operation."
            )
            failure_type = "stagnation"
            confidence = 0.8
        elif dominant_action is not None:
            rule_text = (
                f"Avoid overusing gate {dominant_action}; add exploratory moves when fidelity "
                "improvement slows."
            )
            failure_type = "action_loop"
            confidence = 0.75
        else:
            rule_text = "Use short probing sequences early to estimate promising gate directions."
            failure_type = "insufficient_signal"
            confidence = 0.6

        lesson_id = f"lesson_{trajectory.run_id}_{trajectory.episode_id}"
        lessons.append(
            Lesson(
                lesson_id=lesson_id,
                run_id=trajectory.run_id,
                task_family=trajectory.task_family,
                status=status,
                failure_type=failure_type,
                rule_text=rule_text,
                confidence=float(max(0.0, min(1.0, confidence))),
                supporting_runs=[trajectory.run_id],
            )
        )
    return lessons

