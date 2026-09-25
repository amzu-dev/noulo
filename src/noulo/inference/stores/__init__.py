"""Pluggable vector stores behind the learning memory.

A :class:`VectorStore` persists :class:`MemoryRecord` objects and answers
filtered cosine-similarity searches. All learning logic (dedup, feedback,
blending) lives in ``memory.py``; stores are plain storage and retrieval, so any
local or remote vector database can be plugged in.

Built-in kinds for :func:`open_store`:

- ``"sqlite"`` — stdlib SQLite file (or ``":memory:"``), brute-force cosine in numpy.
- ``"qdrant"`` — embedded (local directory or ``":memory:"``) or remote (``http(s)://`` URL).
- ``"chroma"`` — persistent (local directory), remote (``http(s)://`` URL) or
  ephemeral (``None``).
- ``"package.module:ClassName"`` — any class constructed with the same keyword
  arguments as ``open_store`` (``location``, ``collection``, ``api_key``, ``dim``).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, fields
from typing import Protocol, runtime_checkable

import numpy as np

from noulo.inference.types import Primitive

__all__ = ["MemoryRecord", "VectorStore", "open_store"]


@dataclass(frozen=True, eq=False)
class MemoryRecord:
    """One stored case. ``embedding`` is a float32 vector produced by ``embedder_id``."""

    id: str
    primitive: Primitive
    task: str
    input: str
    input_norm: str
    model_id: str
    embedder_id: str
    embedding: np.ndarray
    observed_value: float | None
    observed_choice: str | None
    verified_value: float | None
    verified_choice: str | None
    hits: int
    created_at: str  # ISO 8601 UTC, fixed width so string order == time order
    updated_at: str

    @property
    def verified(self) -> bool:
        return self.verified_value is not None or self.verified_choice is not None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryRecord):
            return NotImplemented
        return all(
            _field_equal(getattr(self, f.name), getattr(other, f.name)) for f in fields(self)
        )

    __hash__ = None  # type: ignore[assignment]  # mutable-looking payload (ndarray)


def _field_equal(a: object, b: object) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return a is not None and b is not None and np.array_equal(a, b)
    return a == b


@runtime_checkable
class VectorStore(Protocol):
    """Storage and cosine retrieval of memory records. Implementations may be remote."""

    def upsert(self, record: MemoryRecord) -> None:
        """Insert or fully replace the record with ``record.id``."""
        ...

    def get(self, record_id: str) -> MemoryRecord | None: ...

    def find_duplicate(
        self, *, primitive: str, task: str, input_norm: str, model_id: str
    ) -> MemoryRecord | None:
        """A record with exactly these four fields, if any."""
        ...

    def search(
        self,
        *,
        primitive: str,
        task: str,
        embedder_id: str,
        vector: np.ndarray,
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[MemoryRecord, float]]:
        """Records matching the filters with cosine >= min_similarity, best first."""
        ...

    def list(self, *, limit: int, offset: int, primitive: str | None = None) -> list[MemoryRecord]:
        """Records ordered newest ``created_at`` first."""
        ...

    def count(self) -> dict[str, int]:
        """``{"records": total, "verified": records with a verified outcome}``."""
        ...

    def clear(self) -> int:
        """Delete every record; returns how many were deleted."""
        ...

    def close(self) -> None:
        """Release resources. Idempotent."""
        ...


def open_store(
    kind: str,
    *,
    location: str | None,
    collection: str = "noulo_memory",
    api_key: str | None = None,
    dim: int | None = None,
) -> VectorStore:
    """Open a vector store by kind (see module docstring). Unknown kinds raise ValueError."""
    options = {"location": location, "collection": collection, "api_key": api_key, "dim": dim}
    if kind == "sqlite":
        from noulo.inference.stores.sqlite import SqliteVectorStore

        return SqliteVectorStore(**options)
    if kind == "qdrant":
        from noulo.inference.stores.qdrant import QdrantVectorStore

        return QdrantVectorStore(**options)
    if kind == "chroma":
        from noulo.inference.stores.chroma import ChromaVectorStore

        return ChromaVectorStore(**options)
    if ":" in kind:
        return _load_custom(kind)(**options)
    raise ValueError(f"Unknown vector store kind {kind!r}.")


def _load_custom(spec: str) -> type:
    module_name, _, class_name = spec.partition(":")
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(f"Cannot load vector store class {spec!r}.") from exc
