"""Deterministic test doubles implementing the DecisionBackend protocol."""

from __future__ import annotations

import threading
import time

from noulo.inference.types import BackendError, BackendInfo
from noulo.registry import UnknownModelError


class FakeBackend:
    def __init__(
        self,
        id: str = "fake-model",
        *,
        local: bool = True,
        noul_value: float = 0.8,
        choice_probs: list[float] | None = None,
        score_probs: list[float] | None = None,
        delay: float = 0.0,
        error: Exception | None = None,
        warmup_error: Exception | None = None,
        gate: threading.Event | None = None,
    ):
        self.info = BackendInfo(
            id=id,
            backend="onnx-nli" if local else "openai",
            model=id,
            quantization="INT8",
            local=local,
        )
        self.noul_value = noul_value
        self.choice_probs = choice_probs
        self.score_probs = score_probs
        self.delay = delay
        self.error = error
        self.warmup_error = warmup_error
        self.gate = gate
        self.closed = False
        self.warmed = False
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.started = threading.Event()
        self._lock = threading.Lock()

    def _run(self, result):
        if self.closed:
            raise BackendError("backend used after close")
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.gate is not None:
                self.gate.wait(5)
            if self.delay:
                time.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return result
        finally:
            with self._lock:
                self.active -= 1

    def noul(self, input, proposition):
        return self._run(self.noul_value)

    def choice(self, input, question, options):
        probs = self.choice_probs or [1.0 / len(options)] * len(options)
        return self._run(list(probs))

    def score(self, input, question, rubric):
        probs = self.score_probs or [1.0 / len(rubric)] * len(rubric)
        return self._run(list(probs))

    def warmup(self):
        if self.warmup_error is not None:
            raise self.warmup_error
        self.warmed = True

    def close(self):
        self.closed = True


class FakeLoader:
    """Maps model ids to backends; records how often each was loaded."""

    def __init__(self, **backends: FakeBackend):
        self.backends = backends
        self.loads: list[str] = []

    def __call__(self, model_id: str):
        self.loads.append(model_id)
        if model_id not in self.backends:
            raise UnknownModelError(f"Unknown model {model_id!r}")
        return self.backends[model_id]
