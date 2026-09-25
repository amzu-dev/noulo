"""Spec examples end-to-end through the real bundled model, tuned profile and calibration."""

import pytest

from noulo.embedded import Noulo

pytestmark = pytest.mark.model

SEVERITY = ["insignificant", "low", "medium", "high", "critical"]


@pytest.fixture(scope="module")
def engine():
    with Noulo(learning_enabled=False, models_file=None) as n:
        yield n


def test_noul_double_charge(engine):
    value = engine.noul(
        "I checked my account and you have taken the subscription payment twice.",
        "The customer reports being charged more than once.",
    )
    assert value > 0.8


def test_noul_generalises_to_unseen_proposition(engine):
    assert (
        engine.noul(
            "The invoice has remained unpaid for 120 days.", "The customer has an overdue payment."
        )
        > 0.8
    )


def test_noul_contradiction_is_low(engine):
    assert (
        engine.noul("The package arrived on time and undamaged.", "The shipment arrived damaged.")
        < 0.2
    )


def test_choice_routes_double_charge_to_billing(engine):
    assert (
        engine.choice(
            "The customer says their subscription payment was taken twice.",
            "Which department should handle this?",
            {"A": "Billing", "B": "Technical Support", "C": "Sales"},
        )
        == "A"
    )


def test_score_total_outage_is_high(engine):
    assert (
        engine.score(
            "The production system is unavailable for every customer.",
            "How severe is this incident?",
            SEVERITY,
        )
        > 0.6
    )


@pytest.mark.parametrize("value_range", [(0.0, 1.0)])
def test_hard_invariants_hold_on_the_real_model(engine, value_range):
    lo, hi = value_range
    assert lo <= engine.noul("x", "y") <= hi
    assert lo <= engine.score("x", "How much?", ["a", "b", "c"]) <= hi
    assert engine.choice("x", "Which?", {"p": "one", "q": "two"}) in {"p", "q"}
