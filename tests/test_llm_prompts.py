import pytest

from noulo.inference.llm_prompts import (
    NOUL_LABELS,
    SYSTEM_PROMPT,
    choice_task,
    messages,
    noul_task,
    noul_value,
    score_task,
)


def test_noul_task_uses_yes_no_unknown_labels_and_quotes_input():
    task = noul_task("The invoice is late.", "Payment is overdue.")
    assert tuple(task.labels) == NOUL_LABELS == ("yes", "no", "unknown")
    assert '"""\nThe invoice is late.\n"""' in task.prompt
    assert "Proposition: Payment is overdue." in task.prompt


def test_choice_task_letters_options_in_order():
    task = choice_task("x", "Which team?", ["Billing", "Sales", "Support"])
    assert task.labels == "ABC"
    assert "A. Billing\nB. Sales\nC. Support" in task.prompt


def test_score_task_states_rubric_order():
    task = score_task("x", "How bad?", ["low", "high"])
    assert task.labels == "AB"
    assert "lowest (A) to highest (B)" in task.prompt


def test_too_many_candidates_rejected():
    with pytest.raises(ValueError):
        choice_task("x", "q", ["o"] * 27)


def test_noul_value_maps_unknown_to_half():
    assert noul_value([0.0, 0.0, 1.0]) == pytest.approx(0.5)
    assert noul_value([0.6, 0.2, 0.2]) == pytest.approx(0.7)


def test_messages_pair_system_and_user():
    assert messages("hi") == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "hi"},
    ]
