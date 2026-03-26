from __future__ import annotations


def build_student_prompt(always_on: bool = True) -> str:
    if always_on:
        return ""
    return "Solve the user task correctly and efficiently."

