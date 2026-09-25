import math

import numpy as np
import pytest

from noulo.inference.nli_backend import NliBackend
from noulo.inference.types import DecisionBackend, ModelLoadError


class KeywordNliModel:
    """Fake 3-class NLI model: entailment when the hypothesis shares a keyword with the premise."""

    labels = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def __init__(self, logits_override=None):
        self.calls = 0
        self.closed = False
        self._override = logits_override

    def predict_logits(self, pairs):
        self.calls += 1
        rows = []
        for premise, hypothesis in pairs:
            if self._override is not None:
                rows.append(self._override)
                continue
            words = {w.strip(".,?!").lower() for w in premise.split()}
            hits = sum(w.strip(".,?!").lower() in words for w in hypothesis.split())
            rows.append([float(hits), 0.0, -float(hits)])
        return np.array(rows, dtype=np.float32).reshape(len(pairs), 3)

    def close(self):
        self.closed = True


def make_backend(model=None, **kwargs):
    return NliBackend(
        model or KeywordNliModel(), model_id="fake-nli", quantization="INT8", **kwargs
    )


def test_backend_satisfies_protocol():
    assert isinstance(make_backend(), DecisionBackend)


def test_info_describes_model_without_paths():
    info = make_backend().info
    assert (info.id, info.backend, info.quantization, info.local) == (
        "fake-nli",
        "onnx-nli",
        "INT8",
        True,
    )


def test_noul_is_high_when_hypothesis_is_entailed():
    backend = make_backend()
    assert backend.noul("the invoice is overdue", "invoice overdue") > 0.9


def test_choice_returns_normalised_distribution_in_option_order():
    backend = make_backend()
    probs = backend.choice("billing charged twice", "Which team?", ["Sales", "Billing"])
    assert len(probs) == 2
    assert sum(probs) == pytest.approx(1.0)
    assert probs[1] > probs[0]


def test_choice_scores_all_options_in_one_batch():
    model = KeywordNliModel()
    make_backend(model).choice("x", "q?", ["a", "b", "c", "d"])
    assert model.calls == 1


def test_score_returns_distribution_over_levels():
    backend = make_backend()
    probs = backend.score("impact critical outage", "How severe?", ["low", "medium", "critical"])
    assert len(probs) == 3
    assert sum(probs) == pytest.approx(1.0)
    assert int(np.argmax(probs)) == 2


def test_score_temperature_flattens_distribution():
    sharp = make_backend().score("critical", "How severe?", ["low", "critical"])
    flat = make_backend(score_temperature=10.0).score(
        "critical", "How severe?", ["low", "critical"]
    )
    assert max(flat) < max(sharp)


def test_warmup_rejects_non_finite_outputs():
    backend = make_backend(KeywordNliModel(logits_override=[math.nan, 0.0, 0.0]))
    with pytest.raises(ModelLoadError):
        backend.warmup()


def test_close_releases_model():
    model = KeywordNliModel()
    make_backend(model).close()
    assert model.closed


# ------------------------------------------------------------ real model


@pytest.fixture(scope="module")
def real_backend(real_nli_model):
    return NliBackend(real_nli_model, model_id="nli-deberta-v3-xsmall-int8", quantization="INT8")


@pytest.mark.model
def test_real_noul_spec_example_is_strongly_true(real_backend):
    value = real_backend.noul(
        "I checked my account and you have taken the subscription payment twice.",
        "The customer reports being charged more than once.",
    )
    assert value > 0.8


@pytest.mark.model
def test_real_noul_contradiction_is_low(real_backend):
    value = real_backend.noul(
        "The package arrived on time and in perfect condition.",
        "The shipment arrived damaged.",
    )
    assert value < 0.2


@pytest.mark.model
def test_real_choice_spec_example_prefers_billing(real_backend):
    probs = real_backend.choice(
        "The customer says their subscription payment was taken twice.",
        "Which department should handle this?",
        ["Billing", "Technical Support", "Sales"],
    )
    assert int(np.argmax(probs)) == 0


def test_score_method_entailment_minus_contradiction_uses_both_logits():
    class Fixed(KeywordNliModel):
        def predict_logits(self, pairs):
            # level 0: strong entailment but also strong contradiction; level 1: mild entailment
            return np.array([[3.0, 0.0, 3.0], [1.0, 0.0, -2.0]], dtype=np.float32)

    by_entailment = make_backend(Fixed()).score("x", "q?", ["a", "b"])
    by_margin = make_backend(Fixed(), score_method="entailment-contradiction").score(
        "x", "q?", ["a", "b"]
    )
    assert by_entailment[0] > by_entailment[1]
    assert by_margin[1] > by_margin[0]


def test_unknown_score_method_is_rejected():
    with pytest.raises(ValueError):
        make_backend(score_method="vibes")


def test_score_premise_can_include_question():
    seen = []

    class Recording(KeywordNliModel):
        def predict_logits(self, pairs):
            seen.extend(pairs)
            return super().predict_logits(pairs)

    make_backend(Recording(), score_premise="input+question").score(
        "Input text.", "How bad?", ["a", "b"]
    )
    assert all(premise == "Input text. How bad?" for premise, _ in seen)
