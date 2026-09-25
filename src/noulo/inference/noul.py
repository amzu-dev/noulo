"""Noul: P(proposition is true | input), always within [0, 1].

With a 3-class NLI model the neutral mass counts as "undetermined" (0.5), so
entailment -> 1.0, neutral -> 0.5, contradiction -> 0.0. Two-class zero-shot
models (entailment / not_entailment) fall back to P(entailment).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def noul_hypothesis(proposition: str) -> str:
    return proposition.strip()


def noul_from_nli_probs(probs: np.ndarray, labels: Mapping[str, int]) -> float:
    value = float(probs[labels["entailment"]])
    if "neutral" in labels:
        value += 0.5 * float(probs[labels["neutral"]])
    return min(1.0, max(0.0, value))
