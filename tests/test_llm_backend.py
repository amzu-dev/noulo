import pytest

from noulo.inference.llm_backend import LlmBackend
from noulo.inference.llm_prompts import SYSTEM_PROMPT
from noulo.inference.types import DecisionBackend, ModelLoadError


class FakeCausalLM:
    """Returns a fixed label distribution; records the chat messages it was given."""

    def __init__(self, probs=None, mass=0.98):
        self.probs, self.mass = probs, mass
        self.calls = []
        self.closed = False

    def label_distribution(self, messages, labels):
        self.calls.append((messages, list(labels)))
        probs = self.probs or [1.0 / len(labels)] * len(labels)
        return list(probs)[: len(labels)], self.mass

    def close(self):
        self.closed = True


def make(lm=None, **kwargs):
    return LlmBackend(
        lm or FakeCausalLM(), model_id="qwen3-0.6b-q4f16", quantization="INT4", **kwargs
    )


def test_satisfies_backend_protocol_and_describes_itself():
    backend = make()
    assert isinstance(backend, DecisionBackend)
    assert (backend.info.backend, backend.info.local, backend.info.quantization) == (
        "onnx-llm",
        True,
        "INT4",
    )


def test_noul_maps_yes_no_unknown_distribution_to_value():
    backend = make(FakeCausalLM(probs=[0.6, 0.2, 0.2]))
    assert backend.noul("The invoice is late.", "Payment is overdue.") == pytest.approx(0.7)


def test_prompts_use_shared_system_prompt_and_labels():
    lm = FakeCausalLM()
    make(lm).choice("Charged twice", "Which team?", ["Billing", "Sales"])
    messages, labels = lm.calls[0]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "A. Billing\nB. Sales" in messages[1]["content"]
    assert labels == ["A", "B"]


def test_choice_and_score_return_one_probability_per_candidate():
    backend = make(FakeCausalLM(probs=[0.1, 0.7, 0.2]))
    assert backend.choice("x", "q", ["a", "b", "c"]) == pytest.approx([0.1, 0.7, 0.2])
    assert backend.score("x", "q", ["low", "mid", "high"]) == pytest.approx([0.1, 0.7, 0.2])


def test_one_forward_pass_per_request():
    lm = FakeCausalLM()
    make(lm).choice("x", "q", ["a", "b", "c", "d", "e"])
    assert len(lm.calls) == 1


def test_warmup_rejects_models_that_do_not_answer_with_labels():
    with pytest.raises(ModelLoadError, match="labels"):
        make(FakeCausalLM(mass=0.05)).warmup()


def test_warmup_accepts_label_following_models():
    make(FakeCausalLM(mass=0.9)).warmup()


def test_close_releases_model():
    lm = FakeCausalLM()
    make(lm).close()
    assert lm.closed
