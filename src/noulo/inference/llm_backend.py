"""Decision backend over a local causal LLM (Qwen, Gemma, LFM2, ...) on ONNX Runtime.

Each request is one chat-formatted classification prompt and **one forward
pass**: the probabilities of the label tokens at the next position become the
Noul value, the Choice distribution or the Score distribution. Nothing is
generated, so the output contract (0-1 values, supplied IDs only) holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .llm_prompts import Classification, choice_task, messages, noul_task, noul_value, score_task
from .types import BackendInfo, ModelLoadError

MIN_LABEL_MASS = 0.3  # a model that puts less next-token mass on the labels isn't following them


class CausalLM(Protocol):
    def label_distribution(
        self, messages: list[dict[str, str]], labels: Sequence[str]
    ) -> tuple[list[float], float]:
        """Renormalised next-token distribution over labels, plus the raw mass on them."""
        ...

    def close(self) -> None: ...


class LlmBackend:
    def __init__(self, model: CausalLM, *, model_id: str, quantization: str):
        self._model = model
        self.info = BackendInfo(
            id=model_id,
            backend="onnx-llm",
            model=model_id,
            quantization=quantization,
            local=True,
            device=getattr(model, "device", "cpu"),
        )

    def _classify(self, task: Classification) -> tuple[list[float], float]:
        return self._model.label_distribution(messages(task.prompt), task.labels)

    def noul(self, input: str, proposition: str) -> float:
        probs, _ = self._classify(noul_task(input, proposition))
        return noul_value(probs)

    def choice(self, input: str, question: str, options: Sequence[str]) -> list[float]:
        return self._classify(choice_task(input, question, options))[0]

    def score(self, input: str, question: str, rubric: Sequence[str]) -> list[float]:
        return self._classify(score_task(input, question, rubric))[0]

    def warmup(self) -> None:
        _, mass = self._classify(noul_task("The system is ready.", "The system is ready."))
        if mass < MIN_LABEL_MASS:
            raise ModelLoadError(
                f"Model {self.info.id!r} does not answer with the expected labels "
                f"({mass:.0%} of next-token probability); its chat template may be unsupported."
            )

    def close(self) -> None:
        self._model.close()
