"""Model menu shared by `noulo model` and the interactive `/model` picker.

Every row shows download size, quantisation, measured RAM and measured
accuracy, so the choice is informed rather than guessed.
"""

from __future__ import annotations

from typing import Any

from .registry import CURATED_MODELS

OWN_MODEL = "__own__"


def facts(m: dict[str, Any]) -> str:
    """e.g. '87 MB INT8 · RAM 443 MB · Noul 91% · Choice 68% · ✓' (✓ installed, ↓ download)."""
    if m.get("backend") == "openai":
        return f"{m.get('description', 'OpenAI-compatible endpoint')} · remote"
    parts = [
        f"{m.get('sizeMB', '?')} MB {m.get('quantization', '?')}",
        f"RAM {m['ramMB']} MB" if m.get("ramMB") else "RAM n/a",
    ]
    if m.get("noulAccuracy") is not None:
        parts.append(f"Noul {m['noulAccuracy'] * 100:.0f}%")
    if m.get("choiceAccuracy") is not None:
        parts.append(f"Choice {m['choiceAccuracy'] * 100:.0f}%")
    parts.append("✓" if m.get("installed") else "↓")
    return " · ".join(parts)


def menu_items(listing: list[dict[str, Any]], active: str | None) -> list[tuple[str, str, str]]:
    """(value, label, facts): basic models, larger models, small LLMs, yours, plug-in guide."""
    by_id = {m["id"]: m for m in listing}
    curated_ids = {c.id for c in CURATED_MODELS}

    def row(group: str, model_id: str) -> tuple[str, str, str]:
        mark = " (current)" if model_id == active else ""
        return (
            model_id,
            f"{group:14s} {model_id + mark:36s}",
            facts(by_id.get(model_id, {"id": model_id})),
        )

    items = [row(c.label, c.id) for c in CURATED_MODELS]
    items += [row(m.get("label") or "Larger", m["id"]) for m in listing if m.get("tier") == "large"]
    items += [row(m.get("label") or "LLM", m["id"]) for m in listing if m.get("tier") == "llm"]
    items += [
        row("Yours", m["id"])
        for m in listing
        if m["id"] not in curated_ids and m.get("tier") is None
    ]
    items.append(
        (
            OWN_MODEL,
            "Plug in your own model...",
            "OpenAI-compatible endpoint or your own ONNX model",
        )
    )
    return items
