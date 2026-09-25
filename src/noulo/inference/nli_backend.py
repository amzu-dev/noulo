"""Decision backend on top of a local NLI cross-encoder.

Noul scores (input, proposition) directly. Choice and Score turn every
candidate into a hypothesis, score all hypotheses against the input in one
batch and softmax the entailment logits across candidates (standard NLI
zero-shot classification).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import choice as choice_mod
from . import score as score_mod
from .model import _softmax
from .noul import noul_from_nli_probs, noul_hypothesis
from .types import BackendInfo, ModelLoadError


class NliModel(Protocol):
    labels: dict[str, int]

    def predict_logits(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...

    def close(self) -> None: ...


METHODS = ("entailment", "entailment-contradiction")
PREMISES = ("input", "input+question")


@dataclass(frozen=True)
class CandidateStrategy:
    """How candidates (choice options / rubric levels) are scored against the input.

    - template: hypothesis template name (see choice.TEMPLATES / score.TEMPLATES)
    - method: "entailment" softmaxes entailment logits across candidates;
      "entailment-contradiction" softmaxes the entailment-minus-contradiction margin
    - premise: "input" or "input+question" (question appended to the premise)
    - temperature: softmax temperature across candidates
    """

    template: str
    method: str = "entailment"
    premise: str = "input"
    temperature: float = 1.0

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"Unknown scoring method: {self.method!r}")
        if self.premise not in PREMISES:
            raise ValueError(f"Unknown premise mode: {self.premise!r}")
        if not self.temperature > 0:
            raise ValueError("Temperature must be positive.")


class NliBackend:
    def __init__(
        self,
        model: NliModel,
        *,
        model_id: str,
        quantization: str,
        choice_template: str = choice_mod.DEFAULT_TEMPLATE,
        choice_method: str = "entailment",
        choice_premise: str = "input",
        score_template: str = score_mod.DEFAULT_TEMPLATE,
        score_method: str = "entailment",
        score_premise: str = "input",
        score_temperature: float = 1.0,
    ):
        self._model = model
        self._choice = CandidateStrategy(choice_template, choice_method, choice_premise)
        self._score = CandidateStrategy(
            score_template, score_method, score_premise, score_temperature
        )
        self.info = BackendInfo(
            id=model_id, backend="onnx-nli", model=model_id, quantization=quantization, local=True
        )

    def _candidate_distribution(
        self, input: str, question: str, hypotheses: list[str], strategy: CandidateStrategy
    ) -> list[float]:
        premise = input if strategy.premise == "input" else f"{input.strip()} {question.strip()}"
        logits = self._model.predict_logits([(premise, h) for h in hypotheses])
        labels = self._model.labels
        support = logits[:, labels["entailment"]]
        if strategy.method == "entailment-contradiction":
            against = labels.get("contradiction", labels.get("not_entailment"))
            if against is not None:
                support = support - logits[:, against]
        return _softmax(support / strategy.temperature).astype(float).tolist()

    def noul(self, input: str, proposition: str) -> float:
        logits = self._model.predict_logits([(input, noul_hypothesis(proposition))])
        return noul_from_nli_probs(_softmax(logits)[0], self._model.labels)

    def choice(self, input: str, question: str, options: Sequence[str]) -> list[float]:
        hypotheses = choice_mod.choice_hypotheses(question, options, self._choice.template)
        return self._candidate_distribution(input, question, hypotheses, self._choice)

    def score(self, input: str, question: str, rubric: Sequence[str]) -> list[float]:
        hypotheses = score_mod.score_hypotheses(question, rubric, self._score.template)
        return self._candidate_distribution(input, question, hypotheses, self._score)

    def warmup(self) -> None:
        logits = self._model.predict_logits([("The system is ready.", "The system is ready.")])
        if logits.shape[0] != 1 or not np.all(np.isfinite(logits)):
            raise ModelLoadError("Model readiness check produced invalid output.")

    def close(self) -> None:
        self._model.close()
