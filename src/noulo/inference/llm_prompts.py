"""Classification prompts shared by every LLM backend (remote and local).

Each primitive becomes a single-token classification: the candidates are
shown as labels and the backend reads the probability of each label as the
*next token*. The model never generates a number or an explanation.
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are a strict classifier. Read the input and the task, then reply with exactly one "
    "label from the allowed labels and nothing else: no explanation, no punctuation."
)
NOUL_LABELS = ("yes", "no", "unknown")
_LETTERS = string.ascii_uppercase


@dataclass(frozen=True)
class Classification:
    prompt: str
    labels: Sequence[str]


def letter_labels(count: int) -> str:
    if not 1 <= count <= len(_LETTERS):
        raise ValueError(f"Expected between 1 and {len(_LETTERS)} candidates, got {count}.")
    return _LETTERS[:count]


def _listing(labels: str, items: Sequence[str]) -> str:
    return "\n".join(f"{label}. {item}" for label, item in zip(labels, items, strict=True))


def user_prompt(input: str, task: str) -> str:
    return f'Input:\n"""\n{input}\n"""\n\n{task}'


def noul_task(input: str, proposition: str) -> Classification:
    task = (
        f"Proposition: {proposition}\n\n"
        'Answer "yes" if the input supports the proposition, "no" if the input contradicts '
        'it, or "unknown" if the input does not determine it.\n'
        "Reply with exactly one label: yes, no or unknown."
    )
    return Classification(user_prompt(input, task), NOUL_LABELS)


def choice_task(input: str, question: str, options: Sequence[str]) -> Classification:
    labels = letter_labels(len(options))
    task = (
        f"Question: {question}\n\nOptions:\n{_listing(labels, options)}\n\n"
        f"Reply with exactly one letter: {', '.join(labels)}."
    )
    return Classification(user_prompt(input, task), labels)


def score_task(input: str, question: str, rubric: Sequence[str]) -> Classification:
    labels = letter_labels(len(rubric))
    task = (
        f"Question: {question}\n\n"
        f"Rubric levels, ordered from lowest (A) to highest ({labels[-1]}):\n"
        f"{_listing(labels, rubric)}\n\n"
        "Reply with exactly one letter: the level that best fits the input "
        f"({', '.join(labels)})."
    )
    return Classification(user_prompt(input, task), labels)


def noul_value(probs: Sequence[float]) -> float:
    """P(yes) + 0.5 * P(unknown) for a (yes, no, unknown) distribution."""
    p_yes, _p_no, p_unknown = probs
    return min(1.0, max(0.0, p_yes + 0.5 * p_unknown))


def messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
