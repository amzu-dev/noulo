import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from noulo.inference.choice import choice_hypotheses, select_choice
from noulo.inference.noul import noul_from_nli_probs
from noulo.inference.score import expected_score, score_hypotheses

THREE_CLASS = {"entailment": 0, "neutral": 1, "contradiction": 2}
TWO_CLASS = {"entailment": 0, "not_entailment": 1}

finite = st.floats(min_value=-50, max_value=50, allow_nan=False)


# ---------------------------------------------------------------- noul


def test_noul_entailment_maps_to_one():
    assert noul_from_nli_probs(np.array([1.0, 0.0, 0.0]), THREE_CLASS) == pytest.approx(1.0)


def test_noul_contradiction_maps_to_zero():
    assert noul_from_nli_probs(np.array([0.0, 0.0, 1.0]), THREE_CLASS) == pytest.approx(0.0)


def test_noul_neutral_maps_to_half():
    assert noul_from_nli_probs(np.array([0.0, 1.0, 0.0]), THREE_CLASS) == pytest.approx(0.5)


def test_noul_two_class_model_uses_entailment_probability():
    assert noul_from_nli_probs(np.array([0.3, 0.7]), TWO_CLASS) == pytest.approx(0.3)


@given(st.lists(st.floats(min_value=0, max_value=1), min_size=3, max_size=3))
def test_noul_is_always_in_unit_interval(raw):
    probs = np.array(raw) + 1e-9
    probs = probs / probs.sum()
    assert 0.0 <= noul_from_nli_probs(probs, THREE_CLASS) <= 1.0


# ---------------------------------------------------------------- choice


def test_select_choice_returns_highest_probability_id():
    assert select_choice(["A", "B", "C"], [0.2, 0.7, 0.1]) == "B"


def test_select_choice_breaks_ties_by_supplied_order():
    assert select_choice(["x", "y", "z"], [0.4, 0.4, 0.2]) == "x"


def test_select_choice_ignores_nan_scores():
    assert select_choice(["A", "B"], [math.nan, 0.1]) == "B"


def test_select_choice_all_nan_falls_back_to_first_supplied():
    assert select_choice(["A", "B"], [math.nan, math.nan]) == "A"


def test_select_choice_rejects_length_mismatch():
    with pytest.raises(ValueError):
        select_choice(["A", "B"], [1.0])


def test_select_choice_rejects_empty():
    with pytest.raises(ValueError):
        select_choice([], [])


@given(st.lists(finite, min_size=1, max_size=20))
def test_select_choice_always_returns_a_supplied_id(scores):
    ids = [f"id-{i}" for i in range(len(scores))]
    assert select_choice(ids, scores) in ids


def test_choice_hypotheses_mention_each_option_once():
    hyps = choice_hypotheses("Which department should handle this?", ["Billing", "Sales"])
    assert len(hyps) == 2
    assert "Billing" in hyps[0] and "Sales" in hyps[1]


# ---------------------------------------------------------------- score


def test_expected_score_top_level_is_one():
    assert expected_score([0.0, 0.0, 1.0]) == pytest.approx(1.0)


def test_expected_score_bottom_level_is_zero():
    assert expected_score([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_expected_score_is_probability_weighted_mean_level():
    # levels 0..4, expected level = 0.5*3 + 0.5*4 = 3.5 -> 3.5 / 4
    assert expected_score([0, 0, 0, 0.5, 0.5]) == pytest.approx(0.875)


def test_expected_score_renormalises_unnormalised_input():
    assert expected_score([2.0, 2.0]) == pytest.approx(0.5)


def test_expected_score_requires_two_levels():
    with pytest.raises(ValueError):
        expected_score([1.0])


@given(st.lists(st.floats(min_value=0, max_value=1e6), min_size=2, max_size=11))
def test_expected_score_is_always_in_unit_interval(probs):
    assert 0.0 <= expected_score(probs) <= 1.0


def test_score_hypotheses_follow_rubric_order():
    hyps = score_hypotheses("How severe is this incident?", ["low", "medium", "high"])
    assert [("low" in h, "medium" in h, "high" in h) for h in hyps] == [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ]


def test_expected_score_with_no_probability_mass_is_midpoint():
    assert expected_score([0.0, float("nan"), 0.0]) == pytest.approx(0.5)
