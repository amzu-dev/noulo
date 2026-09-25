"""Evaluation dataset loading (JSON Lines with a `split` field)."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_DATA_DIR = Path("benchmark/data")


def load_dataset(path: str | Path, split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if split is None or r.get("split") == split]
