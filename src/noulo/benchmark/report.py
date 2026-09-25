"""Human-readable benchmark report (all numbers are measured, never hard-coded)."""

from __future__ import annotations

import math
from typing import Any

MIB = 1024 * 1024


def _fmt(value: Any, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if kind == "bytes":
        return f"{value / MIB:.1f} MiB"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "ms":
        return f"{value:.1f} ms"
    if kind == "rps":
        return f"{value:.1f} req/s"
    return f"{value:.3f}"


def format_report(r: dict[str, Any]) -> str:
    rows = [
        ("Model", r.get("model", "n/a")),
        ("Quantisation", r.get("quantization", "n/a")),
        ("Model size", _fmt(r.get("modelSizeBytes"), "bytes")),
        ("Peak RAM", _fmt(r.get("peakRssBytes"), "bytes")),
        ("Cold startup", _fmt(r.get("coldStartMs"), "ms")),
        ("Choice accuracy", _fmt(r.get("choiceAccuracy"), "pct")),
        ("Noul accuracy", _fmt(r.get("noulAccuracy"), "pct")),
        (
            "Noul calibration",
            f"ECE {_fmt(r.get('noulEce'), 'num')}, Brier {_fmt(r.get('noulBrier'), 'num')}",
        ),
        ("Score MAE", _fmt(r.get("scoreMae"), "num")),
        ("P50 inference", _fmt(r.get("p50Ms"), "ms")),
        ("P95 inference", _fmt(r.get("p95Ms"), "ms")),
        ("Throughput", _fmt(r.get("throughputRps"), "rps")),
    ]
    return "\n".join(f"{label + ':':24s}{value}" for label, value in rows)


def format_comparison(results: list[dict[str, Any]]) -> str:
    header = (
        "| Model | Quant | Size | Peak RAM | Cold start | Choice acc | Noul acc | Noul ECE "
        "| Score MAE | P50 | P95 |"
    )
    lines = [header, "|" + "---|" * 11]
    for r in results:
        if r.get("error"):
            lines.append(
                f"| {r['model']} | {r.get('quantization', '')} | failed: {r['error']} |" + " |" * 8
            )
            continue
        lines.append(
            f"| {r['model']} | {r.get('quantization')} | {_fmt(r.get('modelSizeBytes'), 'bytes')} "
            f"| {_fmt(r.get('peakRssBytes'), 'bytes')} | {_fmt(r.get('coldStartMs'), 'ms')} "
            f"| {_fmt(r.get('choiceAccuracy'), 'pct')} | {_fmt(r.get('noulAccuracy'), 'pct')} "
            f"| {_fmt(r.get('noulEce'), 'num')} | {_fmt(r.get('scoreMae'), 'num')} "
            f"| {_fmt(r.get('p50Ms'), 'ms')} | {_fmt(r.get('p95Ms'), 'ms')} |"
        )
    return "\n".join(lines)
