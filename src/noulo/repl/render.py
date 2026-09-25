"""Rich renderings of decision results for the interactive session."""

from __future__ import annotations

from typing import Any

from rich.text import Text

BAR_WIDTH = 24


def meter(value: float, style: str) -> Text:
    filled = round(max(0.0, min(1.0, value)) * BAR_WIDTH)
    return Text("━" * filled, style=style) + Text("─" * (BAR_WIDTH - filled), style="dim")


def verdict(value: float) -> tuple[str, str]:
    if value >= 0.65:
        return "Yes", "bold green"
    if value <= 0.35:
        return "No", "bold red"
    return "Uncertain", "bold yellow"


def footer(latency_ms: float, diagnostics: dict[str, Any] | None) -> Text:
    parts = [f"{latency_ms:.0f} ms"]
    if diagnostics:
        parts.append(diagnostics.get("model", ""))
        learning = diagnostics.get("learning") or {}
        if learning.get("applied"):
            parts.append(
                f"learning moved this {learning['influence'] * 100:.0f}% "
                f"({learning['matches']} similar case{'s' if learning['matches'] != 1 else ''})"
            )
    return Text("  " + " · ".join(p for p in parts if p), style="dim")


def probabilities(
    diagnostics: dict[str, Any] | None, labels: dict[str, str] | None = None
) -> list[Text]:
    rows: list[Text] = []
    for key, p in ((diagnostics or {}).get("probabilities") or {}).items():
        name = f"{key} {labels[key]}" if labels and key in labels else key
        rows.append(Text(f"    {name[:28]:28s} ") + meter(p, "cyan") + Text(f" {p:.2f}"))
    return rows


def noul(value: float) -> Text:
    word, style = verdict(value)
    return (
        Text("  ● ", style=style)
        + Text(f"{word:9s}", style=style)
        + Text(f" {value:.2f}  ")
        + meter(value, style)
    )


def choice(choice_id: str, text: str) -> Text:
    return Text("  → ", style="bold cyan") + Text(choice_id, style="bold") + Text(f"  {text}")


def score(value: float, rubric: list[str]) -> Text:
    level = rubric[round(value * (len(rubric) - 1))] if rubric else ""
    return (
        Text("  ◆ ", style="bold magenta")
        + Text(f"{value:.2f}", style="bold")
        + Text(f"  {level}", style="magenta")
        + Text("  ")
        + meter(value, "magenta")
    )
