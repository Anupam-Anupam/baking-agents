from __future__ import annotations

import re
from pathlib import Path

from sales_agent.schemas.types import Lesson
from sales_agent.utils.io import append_jsonl, read_jsonl


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class LessonStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def add_lessons(self, lessons: list[Lesson]) -> None:
        for lesson in lessons:
            append_jsonl(self.db_path, lesson.to_dict())

    def all_lessons(self) -> list[Lesson]:
        rows = read_jsonl(self.db_path)
        return [Lesson(**row) for row in rows]

    def retrieve(self, query: str, top_k: int = 5) -> list[Lesson]:
        q_tokens = _tokenize(query)
        scored: list[tuple[float, Lesson]] = []
        for lesson in self.all_lessons():
            s = _jaccard(q_tokens, _tokenize(lesson.rule_text)) * lesson.confidence
            if s > 0:
                scored.append((s, lesson))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [lesson for _, lesson in scored[:top_k]]

