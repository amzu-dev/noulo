"""DecisionEngine: the single inference implementation behind the REST API,
the CLI and the Python module API.

Responsibilities:
- lifecycle (load once, readiness check, graceful shutdown)
- concurrency control (bounded parallelism + bounded queue)
- runtime model switching (in-flight requests finish on the old model)
- calibration (Noul), learning memory (optional), primitive reduction
- hard output invariants: 0 <= noul, score <= 1; choice is a supplied ID
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .calibration import Calibrator, IdentityCalibrator
from .choice import select_choice
from .score import expected_score
from .types import BackendInfo, DecisionBackend, LearningAdjustment, ModelLoadError, Primitive

log = logging.getLogger(__name__)


class EngineUnavailable(RuntimeError):
    """The engine cannot accept work (not ready / shutting down)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class EngineBusy(RuntimeError):
    """Too many requests are already queued."""


class RecordNotFound(KeyError):
    """No learning record with this id."""


class InferenceError(RuntimeError):
    """The backend failed or violated the output contract."""

    def __init__(self, message: str, *, remote: bool = False):
        super().__init__(message)
        self.remote = remote


@dataclass
class EngineResult:
    type: Primitive
    value: float | str
    record_id: str | None = None
    diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "value": self.value}


@dataclass
class _Slot:
    """An active backend plus its calibrator, reference-counted for safe switching."""

    backend: DecisionBackend
    calibrator: Calibrator
    users: int = 0
    retired: bool = False


@dataclass
class _Outcome:
    value: float | str
    observed: float | str  # model outcome before learning, recorded in memory
    probabilities: dict[str, float] | None = None
    raw: float | None = None
    learning: LearningAdjustment = field(default_factory=LearningAdjustment)


def _check_unit(value: float, what: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise InferenceError(f"Backend returned an invalid {what}.")
    return value


def _check_distribution(probs: Sequence[float], size: int, what: str) -> list[float]:
    if len(probs) != size:
        raise InferenceError(
            f"Backend returned {len(probs)} {what} probabilities, expected {size}."
        )
    return [_check_unit(p, f"{what} probability") for p in probs]


class DecisionEngine:
    def __init__(
        self,
        *,
        load_backend: Callable[[str], DecisionBackend],
        model_id: str,
        load_calibrator: Callable[[str], Calibrator] = lambda _id: IdentityCalibrator(),
        open_memory: Callable[[], Any] | None = None,
        learning: bool = False,
        max_concurrency: int = 1,
        max_queue: int = 32,
    ):
        self._load_backend = load_backend
        self._load_calibrator = load_calibrator
        self._open_memory = open_memory
        self._initial_model = model_id
        self._learning = learning
        self._memory: Any = None
        self._max_concurrency = max(1, max_concurrency)
        self._max_pending = self._max_concurrency + max(0, max_queue)

        self._slots = threading.BoundedSemaphore(self._max_concurrency)
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._switch_lock = threading.Lock()
        self._pending = 0
        self._slot: _Slot | None = None
        self.state = "created"  # created -> starting -> ready -> stopping -> stopped | failed

    # ------------------------------------------------------------------ lifecycle

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @property
    def model_info(self) -> BackendInfo:
        if self._slot is None:
            raise EngineUnavailable("MODEL_NOT_READY", "The model is not loaded.")
        return self._slot.backend.info

    def _load_slot(self, model_id: str) -> _Slot:
        backend = self._load_backend(model_id)
        try:
            backend.warmup()
            calibrator = self._load_calibrator(model_id)
        except Exception as exc:
            backend.close()
            if isinstance(exc, ModelLoadError):
                raise
            raise ModelLoadError(f"Readiness check failed: {type(exc).__name__}") from exc
        return _Slot(backend=backend, calibrator=calibrator)

    def start(self) -> None:
        self.state = "starting"
        try:
            self._slot = self._load_slot(self._initial_model)
            if self._learning:
                self._ensure_memory()
        except Exception:
            self.state = "failed"
            raise
        self.state = "ready"
        log.info("Engine ready with model %s", self._slot.backend.info.id)

    def shutdown(self, timeout: float = 10.0) -> bool:
        """Stop accepting work, wait for in-flight requests, release resources.

        Returns True when all in-flight work finished within `timeout`.
        """
        with self._lock:
            if self.state == "stopped":
                return True
            self.state = "stopping"
            deadline = time.monotonic() + timeout
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._idle.wait(remaining)
            drained = self._pending == 0
            slot, self._slot = self._slot, None
            memory, self._memory = self._memory, None
            self.state = "stopped"
        if slot is not None:
            slot.backend.close()
        if memory is not None:
            memory.close()
        if not drained:
            log.warning("Shutdown timed out with requests still running.")
        return drained

    # ------------------------------------------------------------------ model switching

    def switch_model(self, model_id: str) -> BackendInfo:
        """Load, warm up and atomically activate another model.

        The current model keeps serving until the new one is ready; requests
        already running finish on the old model, which is then released.
        """
        with self._switch_lock:
            if self.state != "ready":
                raise EngineUnavailable("MODEL_NOT_READY", "The engine is not ready.")
            new_slot = self._load_slot(model_id)
            with self._lock:
                old, self._slot = self._slot, new_slot
                if old is not None:
                    old.retired = True
                    close_now = old.users == 0
            if old is not None and close_now:
                old.backend.close()
            log.info("Switched model to %s", new_slot.backend.info.id)
            return new_slot.backend.info

    # ------------------------------------------------------------------ learning

    @property
    def learning_enabled(self) -> bool:
        return self._learning

    @property
    def memory(self) -> Any:
        return self._memory

    def _ensure_memory(self) -> Any:
        if self._memory is None:
            if self._open_memory is None:
                raise EngineUnavailable(
                    "LEARNING_UNAVAILABLE", "Learning memory is not configured."
                )
            self._memory = self._open_memory()
        return self._memory

    def set_learning(self, enabled: bool) -> bool:
        if enabled:
            self._ensure_memory()
        self._learning = enabled
        return self._learning

    def _require_learning(self) -> Any:
        if not self._learning_active():
            raise EngineUnavailable(
                "LEARNING_DISABLED", "Learning is disabled; enable it to record feedback."
            )
        return self._memory

    @staticmethod
    def _as_outcome(expected: Any) -> dict[str, Any]:
        if isinstance(expected, bool):
            return {"value": 1.0 if expected else 0.0}
        if isinstance(expected, str):
            return {"choice_id": expected}
        return {"value": float(expected)}

    def feedback(self, record_id: str, expected: Any) -> str:
        """Attach the correct outcome to a past evaluation (by its record id)."""
        memory = self._require_learning()
        if not memory.feedback(record_id, **self._as_outcome(expected)):
            raise RecordNotFound(record_id)
        return record_id

    def teach(self, request: Any, expected: Any) -> str:
        """Store a verified outcome for a full request without evaluating it first."""
        from ..api.validation import ChoiceRequest, NoulRequest, ScoreRequest
        from .memory import task_key

        memory = self._require_learning()
        if isinstance(request, NoulRequest):
            task = task_key("noul", proposition=request.proposition)
            outcome = self._as_outcome(expected)
            primitive = "noul"
        elif isinstance(request, ChoiceRequest):
            ids = [c.id for c in request.choices]
            if expected not in ids:
                raise ValueError(f"Expected choice must be one of the supplied IDs: {ids}.")
            task = task_key(
                "choice",
                question=request.question,
                options=[(c.id, c.text) for c in request.choices],
            )
            outcome, primitive = {"choice_id": expected}, "choice"
        elif isinstance(request, ScoreRequest):
            if isinstance(expected, str):
                levels = [level.lower() for level in request.rubric]
                if expected.lower() not in levels:
                    raise ValueError(
                        f"Expected level must be one of the rubric levels: {request.rubric}."
                    )
                expected = levels.index(expected.lower()) / (len(levels) - 1)
            task = task_key("score", question=request.question, rubric=list(request.rubric))
            outcome, primitive = self._as_outcome(expected), "score"
        else:
            raise TypeError(f"Unsupported request type: {type(request).__name__}")
        return memory.teach(primitive=primitive, task=task, input=request.input, **outcome)

    # ------------------------------------------------------------------ request plumbing

    def _acquire(self) -> _Slot:
        with self._lock:
            if self.state in ("stopping", "stopped"):
                raise EngineUnavailable("SHUTTING_DOWN", "The engine is shutting down.")
            if self.state != "ready" or self._slot is None:
                raise EngineUnavailable("MODEL_NOT_READY", "The model is not ready.")
            if self._pending >= self._max_pending:
                raise EngineBusy("Too many requests in progress; retry shortly.")
            self._pending += 1
            slot = self._slot
            slot.users += 1
        return slot

    def _release(self, slot: _Slot) -> None:
        with self._lock:
            slot.users -= 1
            close_now = slot.retired and slot.users == 0
            self._pending -= 1
            if self._pending == 0:
                self._idle.notify_all()
        if close_now:
            slot.backend.close()

    def _run(
        self,
        primitive: Primitive,
        compute: Callable[[_Slot], _Outcome],
        diagnostics: bool,
        learn: Callable[[_Outcome], None] | None = None,
    ) -> EngineResult:
        slot = self._acquire()
        try:
            with self._slots:
                try:
                    outcome = compute(slot)
                except (InferenceError, EngineUnavailable):
                    raise
                except Exception as exc:
                    log.exception("Inference failed on %s", slot.backend.info.id)
                    raise InferenceError(
                        "Inference failed.", remote=not slot.backend.info.local
                    ) from exc
        finally:
            self._release(slot)

        record_id = None
        if learn is not None and self._learning_active():
            record_id = self._record(learn, outcome)
        return EngineResult(
            type=primitive,
            value=outcome.value,
            record_id=record_id,
            diagnostics=self._diagnostics(slot, outcome) if diagnostics else None,
        )

    def _record(self, learn: Callable[[_Outcome], str | None], outcome: _Outcome) -> str | None:
        try:
            return learn(outcome)
        except Exception:  # learning must never break inference
            log.exception("Failed to record learning memory")
            return None

    @staticmethod
    def _diagnostics(slot: _Slot, outcome: _Outcome) -> dict[str, Any]:
        adj = outcome.learning
        data: dict[str, Any] = {
            "model": slot.backend.info.id,
            "backend": slot.backend.info.backend,
            "learning": {
                "applied": adj.applied,
                "influence": adj.influence,
                "matches": adj.matches,
            },
        }
        if outcome.raw is not None:
            data["raw"] = outcome.raw
        if outcome.probabilities is not None:
            data["probabilities"] = outcome.probabilities
        return data

    def _learning_active(self) -> bool:
        return self._learning and self._memory is not None

    def _recall(self, primitive: Primitive, task: str | None, input: str) -> list:
        if task is None or not self._learning_active():
            return []
        try:
            return self._memory.recall(primitive=primitive, task=task, input=input)
        except Exception:
            log.exception("Learning recall failed")
            return []

    # ------------------------------------------------------------------ primitives

    def noul(self, input: str, proposition: str, *, diagnostics: bool = False) -> EngineResult:
        task = self._task_key("noul", proposition=proposition)

        def compute(slot: _Slot) -> _Outcome:
            raw = _check_unit(slot.backend.noul(input, proposition), "noul value")
            calibrated = _check_unit(slot.calibrator(raw), "calibrated noul value")
            outcome = _Outcome(value=calibrated, observed=calibrated, raw=raw)
            neighbours = self._recall("noul", task, input)
            if neighbours:
                blended, outcome.learning = self._memory.blend_scalar(calibrated, neighbours)
                outcome.value = _check_unit(blended, "noul value")
            return outcome

        def learn(outcome: _Outcome) -> str:
            return self._memory.record(
                primitive="noul",
                task=task,
                input=input,
                model_id=self.model_info.id,
                value=outcome.observed,
            )

        return self._run("noul", compute, diagnostics, learn)

    def choice(
        self, input: str, question: str, choices: Sequence[Any], *, diagnostics: bool = False
    ) -> EngineResult:
        ids = [c.id for c in choices]
        texts = [c.text for c in choices]
        task = self._task_key("choice", question=question, options=list(zip(ids, texts)))

        def compute(slot: _Slot) -> _Outcome:
            probs = _check_distribution(
                slot.backend.choice(input, question, texts), len(ids), "choice"
            )
            observed = select_choice(ids, probs)
            outcome = _Outcome(value=observed, observed=observed)
            neighbours = self._recall("choice", task, input)
            if neighbours:
                probs, outcome.learning = self._memory.blend_distribution(probs, ids, neighbours)
                probs = _check_distribution(probs, len(ids), "choice")
            outcome.value = select_choice(ids, probs)
            outcome.probabilities = dict(zip(ids, probs))
            return outcome

        def learn(outcome: _Outcome) -> str:
            return self._memory.record(
                primitive="choice",
                task=task,
                input=input,
                model_id=self.model_info.id,
                choice_id=outcome.observed,
            )

        return self._run("choice", compute, diagnostics, learn)

    def score(
        self, input: str, question: str, rubric: Sequence[str], *, diagnostics: bool = False
    ) -> EngineResult:
        task = self._task_key("score", question=question, rubric=list(rubric))

        def compute(slot: _Slot) -> _Outcome:
            probs = _check_distribution(
                slot.backend.score(input, question, rubric), len(rubric), "score"
            )
            model_value = _check_unit(expected_score(probs), "score")
            outcome = _Outcome(value=model_value, observed=model_value, raw=model_value)
            outcome.probabilities = dict(zip(rubric, probs))
            neighbours = self._recall("score", task, input)
            if neighbours:
                blended, outcome.learning = self._memory.blend_scalar(model_value, neighbours)
                outcome.value = _check_unit(blended, "score")
            return outcome

        def learn(outcome: _Outcome) -> str:
            return self._memory.record(
                primitive="score",
                task=task,
                input=input,
                model_id=self.model_info.id,
                value=outcome.observed,
            )

        return self._run("score", compute, diagnostics, learn)

    def evaluate(self, request: Any, *, diagnostics: bool = False) -> EngineResult:
        from ..api.validation import ChoiceRequest, NoulRequest, ScoreRequest

        if isinstance(request, NoulRequest):
            return self.noul(request.input, request.proposition, diagnostics=diagnostics)
        if isinstance(request, ChoiceRequest):
            return self.choice(
                request.input, request.question, request.choices, diagnostics=diagnostics
            )
        if isinstance(request, ScoreRequest):
            return self.score(
                request.input, request.question, request.rubric, diagnostics=diagnostics
            )
        raise TypeError(f"Unsupported request type: {type(request).__name__}")

    def _task_key(self, primitive: Primitive, **parts: Any) -> str | None:
        """Task key for the learning memory, or None when learning is inactive."""
        if not self._learning_active():
            return None
        from .memory import task_key

        return task_key(primitive, **parts)
