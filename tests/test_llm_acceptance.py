"""Spec examples through the full engine with a local 4-bit LLM (skipped if not installed)."""

import pytest

from noulo.embedded import Noulo
from tests.conftest import MODELS_DIR

LLM = "qwen3-0.6b-q4f16"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (MODELS_DIR / LLM / "model.onnx").exists(), reason=f"{LLM} not installed"
    ),
]


@pytest.fixture(scope="module")
def engine():
    with Noulo(model=LLM, learning_enabled=False, models_file=None) as n:
        yield n


def test_llm_backend_is_active(engine):
    assert engine.engine.model_info.backend == "onnx-llm"


def test_llm_choice_routes_double_charge_to_billing(engine):
    assert (
        engine.choice(
            "The customer says their subscription payment was taken twice.",
            "Which department should handle this?",
            {"A": "Billing", "B": "Technical Support", "C": "Sales"},
        )
        == "A"
    )


def test_llm_noul_leans_yes_on_the_spec_example(engine):
    assert (
        engine.noul(
            "I checked my account and you have taken the subscription payment twice.",
            "The customer reports being charged more than once.",
        )
        > 0.5
    )


def test_llm_outputs_obey_the_hard_invariants(engine):
    assert 0.0 <= engine.noul("x", "y") <= 1.0
    assert 0.0 <= engine.score("x", "How much?", ["a", "b", "c"]) <= 1.0
    assert engine.choice("x", "Which?", {"p": "one", "q": "two"}) in {"p", "q"}
