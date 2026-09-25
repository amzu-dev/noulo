"""Shared contracts between the engine, decision backends and the learning memory.

A *decision backend* turns (input, question/proposition, candidates) into raw
probabilities. Everything backend-independent — calibration, learning blend,
choice selection, rubric expectation — lives in the engine and primitive
modules so every backend obeys the same output contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

Primitive = Literal["noul", "choice", "score"]
PRIMITIVES: tuple[Primitive, ...] = ("choice", "score", "noul")


class BackendError(RuntimeError):
    """A decision backend failed to produce a result (model/runtime/upstream failure)."""


class ModelLoadError(RuntimeError):
    """A model could not be loaded or failed its readiness check."""


@dataclass(frozen=True)
class ChoiceOption:
    id: str
    text: str


@dataclass(frozen=True)
class BackendInfo:
    """Public, non-sensitive description of a backend. Never contains paths or secrets."""

    id: str
    backend: Literal["onnx-nli", "onnx-llm", "openai"]
    model: str
    quantization: str
    local: bool
    device: str = "cpu"  # where it runs: cpu, coreml, cuda, directml, rocm, or remote


@runtime_checkable
class DecisionBackend(Protocol):
    """Produces raw (uncalibrated) probabilities for the three primitives.

    Implementations must be deterministic for identical inputs and must never
    return free text. All returned probabilities are finite and in [0, 1];
    list-valued results have one entry per candidate, in the given order, and
    sum to 1.
    """

    info: BackendInfo

    def noul(self, input: str, proposition: str) -> float:
        """Raw P(proposition is true | input). 0.5 means undetermined."""
        ...

    def choice(self, input: str, question: str, options: Sequence[str]) -> list[float]:
        """Probability that each option (by text, same order) is the answer."""
        ...

    def score(self, input: str, question: str, rubric: Sequence[str]) -> list[float]:
        """Probability distribution over ordered rubric levels (lowest first)."""
        ...

    def warmup(self) -> None:
        """Lightweight readiness check; raise ModelLoadError if unusable."""
        ...

    def close(self) -> None:
        """Release model/runtime resources. Idempotent."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Sentence embedder used by the learning memory for similarity retrieval."""

    id: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return float32 array of shape (len(texts), dim), each row L2-normalised."""
        ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Recollection:
    """A similar past case retrieved from the learning memory."""

    record_id: str
    similarity: float  # cosine similarity of inputs, 0..1
    verified: bool  # True when the outcome came from explicit feedback
    value: float | None  # noul/score outcome in [0, 1]
    choice_id: str | None  # choice outcome


@dataclass
class LearningAdjustment:
    """How much the learning memory moved a result (for diagnostics)."""

    applied: bool = False
    influence: float = 0.0  # alpha in [0, max_influence]
    matches: int = 0
    record_id: str | None = None
    neighbours: list[Recollection] = field(default_factory=list)
