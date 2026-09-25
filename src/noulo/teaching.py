"""Teach noulo from a file of labelled examples.

A file is JSON Lines (one example per line; blank lines and `#` comments are
skipped) or a JSON array / `{"examples": [...]}`. Each example is an ordinary
`/api/v1/evaluate` request plus the right answer:

    {"type": "noul", "input": "...", "proposition": "...", "expected": true}
    {"type": "choice", "input": "...", "question": "...", "choices": [...], "expected": "A"}
    {"type": "score", "input": "...", "question": "...", "rubric": [...], "expected": "high"}

The benchmark dataset format (`label` yes/no/unknown, `answer`, `expected_level`)
is accepted as well, so labelled evaluation data can be taught directly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

REQUEST_FIELDS = {
    "noul": ("input", "proposition"),
    "choice": ("input", "question", "choices"),
    "score": ("input", "question", "rubric"),
}
NOUL_LABELS = {"yes": True, "no": False, "unknown": 0.5}


class TeachingError(ValueError):
    """An example file or example can't be used for teaching."""


def load_examples(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    """Read (line number, example) pairs from a .jsonl or .json file."""
    path = Path(path)
    if not path.is_file():
        raise TeachingError(f"Example file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json" or text.lstrip().startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TeachingError(
                f"{path.name}: invalid JSON (line {exc.lineno}): {exc.msg}"
            ) from None
        items = data.get("examples") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise TeachingError(f'{path.name}: expected a JSON array or {{"examples": [...]}}.')
        return [(index + 1, item) for index, item in enumerate(items)]
    examples = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            examples.append((number, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise TeachingError(f"{path.name}: invalid JSON on line {number}: {exc.msg}") from None
    return examples


def _infer_type(item: dict[str, Any]) -> str:
    kind = item.get("type")
    if kind in REQUEST_FIELDS:
        return kind
    if kind is None:
        if "proposition" in item:
            return "noul"
        if "choices" in item:
            return "choice"
        if "rubric" in item:
            return "score"
    raise TeachingError('can\'t tell the type: set "type" to noul, choice or score.')


def _expected(kind: str, item: dict[str, Any]) -> Any:
    if "expected" in item:
        return item["expected"]
    if kind == "noul" and item.get("label") in NOUL_LABELS:
        return NOUL_LABELS[item["label"]]
    if kind == "choice" and "answer" in item:
        return item["answer"]
    if kind == "score" and isinstance(item.get("expected_level"), int):
        rubric = item.get("rubric") or []
        level = item["expected_level"]
        if not 0 <= level < len(rubric):
            raise TeachingError(f"expected_level {level} is outside the rubric.")
        return rubric[level]
    raise TeachingError('no answer: add "expected" (true/false or 0-1, a choice ID, or a level).')


def to_request(item: Any) -> tuple[dict[str, Any], Any]:
    """Split one example into an /evaluate request and the expected answer."""
    if not isinstance(item, dict):
        raise TeachingError("each example must be a JSON object.")
    kind = _infer_type(item)
    request = {
        "type": kind,
        **{field: item[field] for field in REQUEST_FIELDS[kind] if field in item},
    }
    return request, _expected(kind, item)


def batches(items: Iterable[dict[str, Any]], max_bytes: int) -> Iterator[list[dict[str, Any]]]:
    """Group items so each `{"items": [...]}` body stays under max_bytes (order kept)."""
    batch: list[dict[str, Any]] = []
    size = len('{"items": []}')
    for item in items:
        item_size = len(json.dumps(item)) + 2
        if batch and size + item_size > max_bytes:
            yield batch
            batch, size = [], len('{"items": []}')
        batch.append(item)
        size += item_size
    if batch:
        yield batch
