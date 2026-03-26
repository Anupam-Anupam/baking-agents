from __future__ import annotations


def build_student_prompt(always_on: bool = True) -> str:
    if always_on:
        return ""
    return "Complete the user's shopping task accurately and efficiently."

