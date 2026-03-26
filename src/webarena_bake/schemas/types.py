from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WebArenaRunRecord:
    run_id: str
    task_id: str
    config_file: str
    task_intent: str
    start_url: str
    model_name: str
    success: bool
    score: float
    steps: int
    terminated: bool
    error: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Lesson:
    lesson_id: str
    run_id: str
    task_id: str
    task_family: str
    status: str
    failure_type: str
    rule_text: str
    confidence: float
    supporting_runs: list[str]
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DistilledRule:
    rule_id: str
    task_family: str
    rule_text: str
    support_count: int
    avg_confidence: float
    contradiction_count: int
    source_lesson_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptRecipe:
    recipe_version: str
    teacher_prompt_name: str
    student_prompt_name: str
    teacher_prompt_text: str
    student_prompt_text: str
    rule_ids: list[str]
    task_families: list[str]
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BakeLineageRecord:
    bake_id: str
    repo_name: str
    base_model: str
    teacher_prompt_version: str
    stim_version: str
    rollout_version: str
    recipe_hash: str
    stim_hash: str
    rollout_hash: str
    created_at: str = field(default_factory=utc_now_iso)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

