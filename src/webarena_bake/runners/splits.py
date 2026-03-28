from __future__ import annotations

import random
from pathlib import Path

from webarena_bake.utils.io import read_json, write_json


def is_shopping_task(config_payload: dict) -> bool:
    intent = str(config_payload.get("intent", "")).lower()
    start_url = str(config_payload.get("start_url", "")).lower()
    site = str(config_payload.get("sites", "")).lower()
    keywords = ("shopping", "shop", "ecommerce", "e-commerce", "product", "cart", "checkout")
    if "shopping" in start_url or "shopping_admin" in start_url:
        return True
    if any(word in intent for word in keywords):
        return True
    return "shopping" in site


def build_shopping_split(
    config_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    configs = sorted(config_dir.glob("*.json"))
    task_ids: list[str] = []
    for cfg in configs:
        payload = read_json(cfg)
        if not isinstance(payload, dict):
            # Skip aggregate files like test.json that contain a list.
            continue
        if is_shopping_task(payload):
            task_id = str(payload.get("task_id", cfg.stem))
            task_ids.append(task_id)

    task_ids = sorted(set(task_ids))
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    split_idx = int(len(task_ids) * train_ratio)
    train_ids = sorted(task_ids[:split_idx])
    val_ids = sorted(task_ids[split_idx:])

    manifest = {
        "seed": seed,
        "train_ratio": train_ratio,
        "total": len(task_ids),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    write_json(output_dir / "shopping_split_manifest.json", manifest)
    write_json(output_dir / "train_ids.json", {"task_ids": train_ids})
    write_json(output_dir / "val_ids.json", {"task_ids": val_ids})
    return manifest

