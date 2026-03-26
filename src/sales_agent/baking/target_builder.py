from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TargetSpec:
    repo_name: str
    target_name: str
    teacher_prompt_name: str
    student_prompt_name: str
    task_family: str

