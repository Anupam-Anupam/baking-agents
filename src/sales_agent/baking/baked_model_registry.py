from __future__ import annotations

from pathlib import Path

from sales_agent.schemas.types import BakeLineageRecord
from sales_agent.utils.io import read_json, write_json


class BakedModelRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path

    def add_record(self, record: BakeLineageRecord) -> None:
        if self.registry_path.exists():
            payload = read_json(self.registry_path)
        else:
            payload = {"bakes": []}
        payload.setdefault("bakes", []).append(record.to_dict())
        write_json(self.registry_path, payload)

