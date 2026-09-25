"""Learning memory: stores evaluated cases and feedback, and nudges future results.

See ``docs/learning.md``. Each record carries the
primitive, a normalised *task key*, the input text and its embedding, the
observed model outcome and optionally a verified outcome from feedback. Similar
past cases for the same task are recalled by cosine similarity and blended into
new results. Storage is delegated to a pluggable
:class:`~noulo.inference.stores.VectorStore` (SQLite, Qdrant, Chroma
or custom); all learning rules live here so every store behaves the same.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import numpy as np

from noulo.inference.stores import MemoryRecord, VectorStore
from noulo.inference.types import (
    ChoiceOption,
    Embedder,
    LearningAdjustment,
    Primitive,
    Recollection,
)

OptionsLike = Mapping[str, str] | Iterable[ChoiceOption] | Iterable[tuple[str, str]]


def _normalise(text: str) -> str:
    """Lowercase, trim and collapse all whitespace runs to single spaces."""
    return " ".join(text.lower().split())


def _option_pairs(options: OptionsLike) -> list[tuple[str, str]]:
    """``(id, text)`` pairs from a mapping, ChoiceOption objects or tuples."""
    if isinstance(options, Mapping):
        return list(options.items())
    return [(o.id, o.text) if isinstance(o, ChoiceOption) else (o[0], o[1]) for o in options]


def _required(text: str | None, name: str, primitive: str) -> str:
    normalised = _normalise(text) if text is not None else ""
    if not normalised:
        raise ValueError(f"{primitive} task key requires a non-empty {name}.")
    return normalised


def task_key(
    primitive: Primitive,
    *,
    proposition: str | None = None,
    question: str | None = None,
    options: OptionsLike | None = None,
    rubric: Sequence[str] | None = None,
) -> str:
    """Normalised identity of a task; only records with equal keys are comparable.

    Noul: the proposition. Choice: the question plus ``id=text`` options sorted
    (so option order does not matter). Score: the question plus the rubric in
    its given order (order carries meaning). Raises ValueError if a part is missing.
    """
    if primitive == "noul":
        return f"noul|{_required(proposition, 'proposition', primitive)}"
    if primitive == "choice":
        pairs = _option_pairs(options) if options is not None else []
        if not pairs:
            raise ValueError("choice task key requires at least one option.")
        entries = sorted(f"{oid.strip()}={_normalise(text)}" for oid, text in pairs)
        return f"choice|{_required(question, 'question', primitive)}|{';'.join(entries)}"
    if primitive == "score":
        if not rubric:
            raise ValueError("score task key requires a non-empty rubric.")
        levels = ";".join(_normalise(level) for level in rubric)
        return f"score|{_required(question, 'question', primitive)}|{levels}"
    raise ValueError(f"Unknown primitive {primitive!r}.")


@dataclass(frozen=True)
class MemoryConfig:
    """Retrieval and blending parameters (see the design doc for the formulas)."""

    top_k: int = 8
    min_similarity: float = 0.80
    feedback_weight: float = 1.0
    observed_weight: float = 0.25
    max_influence: float = 0.9
    prior_strength: float = 0.5


DEFAULT_CONFIG = MemoryConfig()


def _outcome(
    primitive: Primitive, value: float | None, choice_id: str | None
) -> tuple[float | None, str | None]:
    """Validate an outcome for ``primitive``; returns the ``(value, choice_id)`` to store."""
    if primitive in ("noul", "score"):
        if choice_id is not None or value is None or isinstance(value, bool):
            raise ValueError(f"A {primitive} outcome is a value in [0, 1] (no choice_id).")
        number = float(value)
        if math.isnan(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"A {primitive} outcome must be in [0, 1], got {value!r}.")
        return number, None
    if primitive == "choice":
        if value is not None or not isinstance(choice_id, str) or not choice_id.strip():
            raise ValueError("A choice outcome is a non-empty choice_id (no value).")
        return None, choice_id
    raise ValueError(f"Unknown primitive {primitive!r}.")


class _Clock:
    """Strictly increasing ISO 8601 UTC timestamps (fixed width, so they sort as text)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = datetime.min.replace(tzinfo=timezone.utc)

    def now(self) -> str:
        with self._lock:
            current = datetime.now(timezone.utc)
            if current <= self._last:
                current = self._last + timedelta(microseconds=1)
            self._last = current
        return current.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


_clock = _Clock()

TAUGHT_MODEL_ID = "feedback"  # model_id of records created by teach()


class LearningMemory:
    """Records evaluated cases and feedback, recalls similar ones and blends them in.

    Store-agnostic: persistence and similarity search are delegated to a
    :class:`~noulo.inference.stores.VectorStore`. The memory takes
    ownership of the store and embedder and closes both in :meth:`close`. All
    store access is serialised by a lock (callers run on worker threads);
    embedding happens outside the lock.
    """

    def __init__(
        self, store: VectorStore, embedder: Embedder, config: MemoryConfig = DEFAULT_CONFIG
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config
        self._lock = threading.Lock()
        self._closed = False

    def record(
        self,
        *,
        primitive: Primitive,
        task: str,
        input: str,
        model_id: str,
        value: float | None = None,
        choice_id: str | None = None,
    ) -> str:
        """Store an observed (model-produced) outcome; returns the record id.

        The same primitive + task + normalised input + model updates the existing
        record (hit count, observed outcome) and never touches a verified outcome.
        """
        outcome = _outcome(primitive, value, choice_id)
        return self._upsert_case(primitive, task, input, model_id, observed=outcome)

    def teach(
        self,
        *,
        primitive: Primitive,
        task: str,
        input: str,
        value: float | None = None,
        choice_id: str | None = None,
    ) -> str:
        """Store a verified outcome directly (no model output); returns the record id."""
        outcome = _outcome(primitive, value, choice_id)
        return self._upsert_case(primitive, task, input, TAUGHT_MODEL_ID, verified=outcome)

    def feedback(
        self, record_id: str, *, value: float | None = None, choice_id: str | None = None
    ) -> bool:
        """Attach a verified outcome to a record; False if the id is unknown."""
        with self._lock:
            existing = self.store.get(record_id)
            if existing is None:
                return False
            value, choice_id = _outcome(existing.primitive, value, choice_id)
            self.store.upsert(
                replace(
                    existing,
                    verified_value=value,
                    verified_choice=choice_id,
                    updated_at=_clock.now(),
                )
            )
        return True

    def recall(self, *, primitive: Primitive, task: str, input: str) -> list[Recollection]:
        """Most similar past cases for the same primitive, task and embedder."""
        query = self._embed(input)
        with self._lock:
            matches = self.store.search(
                primitive=primitive,
                task=task,
                embedder_id=self.embedder.id,
                vector=query,
                top_k=self.config.top_k,
                min_similarity=self.config.min_similarity,
            )
        return [_recollection(record, similarity) for record, similarity in matches]

    def blend_scalar(
        self, model_value: float, recollections: Sequence[Recollection]
    ) -> tuple[float, LearningAdjustment]:
        """Pull a Noul/Score value toward the weighted mean of similar outcomes."""
        used = [(r, w) for r in recollections if r.value is not None and (w := self._weight(r)) > 0]
        total = sum(w for _, w in used)
        if total <= 0:
            return float(model_value), LearningAdjustment()
        alpha = self._influence(total)
        mean = sum(w * r.value for r, w in used) / total
        blended = min(1.0, max(0.0, (1.0 - alpha) * model_value + alpha * mean))
        return blended, _adjustment(alpha, [r for r, _ in used])

    def blend_distribution(
        self,
        probs: Sequence[float],
        option_ids: Sequence[str],
        recollections: Sequence[Recollection],
    ) -> tuple[list[float], LearningAdjustment]:
        """Mix a Choice distribution with weighted votes of similar outcomes.

        Only outcomes naming one of ``option_ids`` count, so learning can never
        introduce an option that was not supplied.
        """
        if len(probs) != len(option_ids):
            raise ValueError("probs and option_ids must have the same length.")
        index: dict[str, int] = {}
        for i, option_id in enumerate(option_ids):
            index.setdefault(option_id, i)
        votes = np.zeros(len(option_ids))
        used: list[Recollection] = []
        for r in recollections:
            if r.choice_id in index and (w := self._weight(r)) > 0:
                votes[index[r.choice_id]] += w
                used.append(r)
        total = float(votes.sum())
        if total <= 0:
            return [float(p) for p in probs], LearningAdjustment()
        alpha = self._influence(total)
        mixed = (1.0 - alpha) * np.asarray(probs, dtype=float) + alpha * votes / total
        mixed = np.clip(mixed, 0.0, None)
        return (mixed / mixed.sum()).tolist(), _adjustment(alpha, used)

    def records(
        self, *, limit: int = 50, offset: int = 0, primitive: Primitive | None = None
    ) -> list[dict]:
        """Stored cases, newest first, as JSON-friendly dicts (no embeddings)."""
        with self._lock:
            stored = self.store.list(limit=limit, offset=offset, primitive=primitive)
        return [_public(record) for record in stored]

    def stats(self) -> dict:
        """``{"records", "verified", "embedder"}`` for diagnostics."""
        with self._lock:
            counts = self.store.count()
        return {
            "records": counts["records"],
            "verified": counts["verified"],
            "embedder": self.embedder.id,
        }

    def clear(self) -> int:
        """Delete every stored case; returns how many were deleted."""
        with self._lock:
            return self.store.clear()

    def close(self) -> None:
        """Close the store and the embedder. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.store.close()
            self.embedder.close()

    def _upsert_case(
        self,
        primitive: Primitive,
        task: str,
        input: str,
        model_id: str,
        *,
        observed: tuple[float | None, str | None] | None = None,
        verified: tuple[float | None, str | None] | None = None,
    ) -> str:
        """Insert a case, or update the duplicate (same primitive/task/input/model).

        Only the supplied outcome kind is written, so an observation never
        clears or overwrites a verified outcome.
        """
        embedding = self._embed(input)
        input_norm = _normalise(input)
        changes: dict = {"embedding": embedding, "embedder_id": self.embedder.id}
        if observed is not None:
            changes["observed_value"], changes["observed_choice"] = observed
        if verified is not None:
            changes["verified_value"], changes["verified_choice"] = verified
        with self._lock:
            now = _clock.now()
            existing = self.store.find_duplicate(
                primitive=primitive, task=task, input_norm=input_norm, model_id=model_id
            )
            if existing is not None:
                record = replace(existing, hits=existing.hits + 1, updated_at=now, **changes)
            else:
                blank = dict.fromkeys(
                    ("observed_value", "observed_choice", "verified_value", "verified_choice")
                )
                record = MemoryRecord(
                    id=uuid.uuid4().hex,
                    primitive=primitive,
                    task=task,
                    input=input,
                    input_norm=input_norm,
                    model_id=model_id,
                    hits=1,
                    created_at=now,
                    updated_at=now,
                    **{**blank, **changes},
                )
            self.store.upsert(record)
        return record.id

    def _weight(self, recollection: Recollection) -> float:
        kind = self.config.feedback_weight if recollection.verified else self.config.observed_weight
        return max(0.0, recollection.similarity) * kind

    def _influence(self, total_weight: float) -> float:
        """alpha = max_influence * W / (W + prior_strength)."""
        return (
            self.config.max_influence * total_weight / (total_weight + self.config.prior_strength)
        )

    def _embed(self, text: str) -> np.ndarray:
        return np.asarray(self.embedder.embed([text])[0], dtype=np.float32)


def _recollection(record: MemoryRecord, similarity: float) -> Recollection:
    verified = record.verified
    return Recollection(
        record_id=record.id,
        similarity=min(1.0, max(0.0, float(similarity))),
        verified=verified,
        value=record.verified_value if verified else record.observed_value,
        choice_id=record.verified_choice if verified else record.observed_choice,
    )


def _adjustment(alpha: float, used: list[Recollection]) -> LearningAdjustment:
    return LearningAdjustment(applied=True, influence=alpha, matches=len(used), neighbours=used)


def _public(record: MemoryRecord) -> dict:
    return {
        "id": record.id,
        "primitive": record.primitive,
        "task": record.task,
        "input": record.input,
        "observedValue": record.observed_value,
        "observedChoice": record.observed_choice,
        "verifiedValue": record.verified_value,
        "verifiedChoice": record.verified_choice,
        "hits": record.hits,
        "modelId": record.model_id,
        "createdAt": record.created_at,
        "updatedAt": record.updated_at,
    }
