"""Evaluation dataset loading (JSON Lines with a `split` field)."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_DATA_DIR = Path("benchmark/data")


def load_dataset(path: str | Path, split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    groups: dict[str, str | None] = {}
    for row in rows:
        group = row.get("group")
        if group is None:
            continue
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Dataset group must be a non-empty string.")
        group = group.strip()
        row_split = row.get("split")
        if group in groups and groups[group] != row_split:
            raise ValueError(f"Dataset group {group!r} appears in multiple splits.")
        groups[group] = row_split
    return [r for r in rows if split is None or r.get("split") == split]
