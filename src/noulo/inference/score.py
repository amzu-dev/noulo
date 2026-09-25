"""Score: place the input on an ordered rubric and return a value in [0, 1].

The value is the probability-weighted expected level divided by the highest
level index, so a certain "lowest" is 0.0 and a certain "highest" is 1.0.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

TEMPLATES = {
    "question-answer": "{question} {level}.",
    "answer-is": 'The answer to "{question}" is {level}.',
    "level-only": "{level}.",
}
DEFAULT_TEMPLATE = "question-answer"


def score_hypotheses(
    question: str, rubric: Sequence[str], template: str = DEFAULT_TEMPLATE
) -> list[str]:
    fmt = TEMPLATES[template]
    return [fmt.format(question=question.strip(), level=level.strip()) for level in rubric]


def expected_score(probs: Sequence[float]) -> float:
    p = np.asarray(probs, dtype=float)
    if p.size < 2:
        raise ValueError("A rubric needs at least two levels.")
    p = np.where(np.isfinite(p) & (p > 0), p, 0.0)
    total = p.sum()
    if total <= 0:
        return 0.5
    expected_level = float(np.dot(p / total, np.arange(p.size)))
    return min(1.0, max(0.0, expected_level / (p.size - 1)))
