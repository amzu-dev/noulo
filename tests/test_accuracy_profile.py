"""Regression coverage for the calibration-selected default Choice premise."""

import json

import numpy as np

from noulo.inference.nli_backend import NliBackend
from noulo.registry import PROFILES_DIR


def test_default_choice_does_not_append_question_to_input():
    pairs_seen = []

    class RecordingModel:
        labels = {"entailment": 0, "neutral": 1, "contradiction": 2}

        def predict_logits(self, pairs):
            pairs_seen.extend(pairs)
            return np.array([[1.0, 0.0, -1.0]] * len(pairs), dtype=np.float32)

        def close(self):
            pass

    profile = json.loads((PROFILES_DIR / "nli-deberta-v3-xsmall-int8/profile.json").read_text())
    backend = NliBackend(
        RecordingModel(), model_id="nli-deberta-v3-xsmall-int8", quantization="INT8", **profile
    )
    text, question = "A software failure was reported.", "Which team should handle this?"
    backend.choice(text, question, ["Billing", "Technical Support"])
    # The question belongs in the hypothesis, not in the input evidence as well.
    assert all(premise == text for premise, _ in pairs_seen)
    assert all(question in hypothesis for _, hypothesis in pairs_seen)
