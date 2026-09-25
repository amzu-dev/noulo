"""Choice: select exactly one of the supplied options.

`select_choice` is the single place that turns per-option probabilities into
an answer, so every backend obeys the rule that the result is a supplied ID.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

TEMPLATES = {
    "question-answer": "{question} {option}.",
    "answer-is": 'The answer to "{question}" is {option}.',
    "option-only": "{option}.",
}
DEFAULT_TEMPLATE = "question-answer"


def choice_hypotheses(
    question: str, options: Sequence[str], template: str = DEFAULT_TEMPLATE
) -> list[str]:
    fmt = TEMPLATES[template]
    return [fmt.format(question=question.strip(), option=option.strip()) for option in options]


def select_choice(ids: Sequence[str], scores: Sequence[float]) -> str:
    """Highest score wins; ties go to the earliest supplied option; NaN never wins."""
    if not ids:
        raise ValueError("At least one choice is required.")
    if len(ids) != len(scores):
        raise ValueError("Each choice needs exactly one score.")
    best_index, best_score = 0, -math.inf
    for index, score in enumerate(scores):
        score = float(score)
        if not math.isnan(score) and score > best_score:
            best_index, best_score = index, score
    return ids[best_index]
