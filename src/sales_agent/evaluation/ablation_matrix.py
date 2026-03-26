from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    description: str


ABLATION_VARIANTS = [
    Variant("baseline", "Plain prompted model"),
    Variant("retrieval", "Prompted model + atomic lesson retrieval"),
    Variant("distilled_prompt", "Prompted model + distilled batch rules"),
    Variant("baked", "Bread-baked model without retrieval"),
    Variant("baked_plus_retrieval", "Bread-baked model with lightweight retrieval"),
]

