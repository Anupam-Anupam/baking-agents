from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    description: str


ABLATION_VARIANTS = [
    Variant("baseline", "Base model only"),
    Variant("retrieval_only", "Base model + lesson retrieval"),
    Variant("distilled_prompt", "Base model + distilled prompt"),
    Variant("baked", "Baked model only"),
    Variant("baked_plus_retrieval", "Baked model + retrieval"),
]

