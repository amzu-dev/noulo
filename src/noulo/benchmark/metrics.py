"""Plain-numpy evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def accuracy(predicted: Sequence, expected: Sequence) -> float:
    if not expected:
        return float("nan")
    return float(np.mean([p == e for p, e in zip(predicted, expected)]))


def brier_score(probs: Sequence[float], labels: Sequence[float]) -> float:
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


def log_loss(probs: Sequence[float], labels: Sequence[float]) -> float:
    p = np.clip(np.asarray(probs, float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if p.size else float("nan")


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[float], bins: int = 10
) -> float:
    """Weighted mean |confidence - observed frequency| over equal-width bins."""
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    if not p.size:
        return float("nan")
    index = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def mean_absolute_error(predicted: Sequence[float], expected: Sequence[float]) -> float:
    p, e = np.asarray(predicted, float), np.asarray(expected, float)
    return float(np.mean(np.abs(p - e))) if p.size else float("nan")


def percentile_ms(seconds: Sequence[float], q: float) -> float:
    return (
        float(np.percentile(np.asarray(seconds, float), q) * 1000.0)
        if len(seconds)
        else float("nan")
    )
