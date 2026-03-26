from __future__ import annotations

from pathlib import Path

from webarena_bake.schemas.types import BakeLineageRecord
from webarena_bake.utils.io import read_json, write_json


class BakedModelRegistry:
    def __init__(self, path: Path):
        self.path = path

    def add_record(self, record: BakeLineageRecord) -> None:
        payload = {"bakes": []}
        if self.path.exists():
            payload = read_json(self.path)
        payload.setdefault("bakes", []).append(record.to_dict())
        write_json(self.path, payload)

