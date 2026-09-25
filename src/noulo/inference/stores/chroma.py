"""Chroma vector store: persistent (local directory), remote (URL) or ephemeral.

The collection uses cosine space (``{"hnsw:space": "cosine"}``), so the
returned distance is ``1 - cosine`` and similarity is ``1 - distance``. The
input text is the Chroma document; every other field is metadata.

Metadata cannot *store* ``None``. On upsert Chroma merges metadata and treats
``None`` as "remove this key", so every field is always sent (nulls as
``None``) to get true replace semantics, and a missing key decodes to ``None``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from noulo.inference.stores import MemoryRecord

try:
    import chromadb
except ImportError as exc:
    raise ImportError(
        "The 'chroma' vector store needs chromadb: pip install 'noulo[chroma]'"
    ) from exc

_METADATA_FIELDS = (
    "id",
    "primitive",
    "task",
    "input_norm",
    "model_id",
    "embedder_id",
    "observed_value",
    "observed_choice",
    "verified_value",
    "verified_choice",
    "hits",
    "created_at",
    "updated_at",
)
_INCLUDE = ["metadatas", "documents", "embeddings"]


def _metadata(record: MemoryRecord) -> dict[str, Any]:
    metadata: dict[str, Any] = {name: getattr(record, name) for name in _METADATA_FIELDS}
    for name in ("observed_value", "verified_value"):
        if metadata[name] is not None:
            metadata[name] = float(metadata[name])
    metadata["verified"] = record.verified
    return metadata


def _record(metadata: dict[str, Any], document: str | None, embedding: Any) -> MemoryRecord:
    values = {name: metadata.get(name) for name in _METADATA_FIELDS}
    values["hits"] = int(values["hits"])
    return MemoryRecord(
        input=document or "", embedding=np.asarray(embedding, dtype=np.float32), **values
    )


def _records(result: dict[str, Any]) -> list[MemoryRecord]:
    return [
        _record(metadata, document, embedding)
        for metadata, document, embedding in zip(
            result["metadatas"], result["documents"], result["embeddings"], strict=True
        )
    ]


def _where(**conditions: Any) -> dict[str, Any] | None:
    """Chroma ``where`` clause; ``$and`` requires at least two expressions."""
    clauses = [{key: value} for key, value in conditions.items()]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _client(location: str | None, api_key: str | None) -> Any:
    if location is None:
        return chromadb.EphemeralClient()
    if location.startswith(("http://", "https://")):
        url = urlparse(location)
        ssl = url.scheme == "https"
        return chromadb.HttpClient(
            host=url.hostname or "localhost",
            port=url.port or (443 if ssl else 8000),
            ssl=ssl,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        )
    Path(location).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=location)


class ChromaVectorStore:
    """Vector store backed by a Chroma collection."""

    def __init__(
        self,
        *,
        location: str | None = None,
        collection: str = "noulo_memory",
        api_key: str | None = None,
        dim: int | None = None,
    ) -> None:
        # ``dim`` is accepted for factory symmetry; Chroma fixes it on first insert.
        self._ephemeral = location is None
        if self._ephemeral:
            # Ephemeral clients share one in-process system; a private collection
            # gives each store ":memory:"-style isolation.
            collection = f"{collection}-{uuid.uuid4().hex[:12]}"
        self._client = _client(location, api_key)
        self._name = collection
        self._collection = self._open_collection()
        self._dim = self._existing_dim()
        self._closed = False

    def upsert(self, record: MemoryRecord) -> None:
        vector = np.asarray(record.embedding, dtype=np.float32)
        if self._dim is not None and len(vector) != self._dim:
            raise ValueError(
                f"Collection '{self._name}' holds {self._dim}-d vectors, got {len(vector)}; "
                "use a separate collection per embedder width."
            )
        self._collection.upsert(
            ids=[record.id],
            embeddings=[vector],
            documents=[record.input],
            metadatas=[_metadata(record)],
        )
        self._dim = len(vector)

    def get(self, record_id: str) -> MemoryRecord | None:
        found = _records(self._collection.get(ids=[record_id], include=_INCLUDE))
        return found[0] if found else None

    def find_duplicate(
        self, *, primitive: str, task: str, input_norm: str, model_id: str
    ) -> MemoryRecord | None:
        where = _where(primitive=primitive, task=task, input_norm=input_norm, model_id=model_id)
        found = _records(self._collection.get(where=where, limit=1, include=_INCLUDE))
        return found[0] if found else None

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
        query = np.asarray(vector, dtype=np.float32)
        if self._dim is None or len(query) != self._dim or top_k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[query],
            n_results=top_k,
            where=_where(primitive=primitive, task=task, embedder_id=embedder_id),
            include=[*_INCLUDE, "distances"],
        )
        records = _records({key: result[key][0] for key in _INCLUDE})
        similarities = [1.0 - float(distance) for distance in result["distances"][0]]
        return [
            (record, similarity)
            for record, similarity in zip(records, similarities, strict=True)
            if similarity >= min_similarity
        ]

    def list(self, *, limit: int, offset: int, primitive: str | None = None) -> list[MemoryRecord]:
        # Chroma has no server-side ordering, so sort by creation time client-side.
        where = _where(primitive=primitive) if primitive else None
        records = _records(self._collection.get(where=where, include=_INCLUDE))
        records.sort(key=lambda r: (r.created_at, r.id), reverse=True)
        return records[offset : offset + limit]

    def count(self) -> dict[str, int]:
        verified = self._collection.get(where={"verified": True}, include=[])["ids"]
        return {"records": self._collection.count(), "verified": len(verified)}

    def clear(self) -> int:
        removed = self._collection.count()
        self._client.delete_collection(self._name)
        self._collection = self._open_collection()
        self._dim = None
        return removed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ephemeral:
            self._client.delete_collection(self._name)
        self._client.close()

    def _open_collection(self) -> Any:
        return self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})

    def _existing_dim(self) -> int | None:
        sample = self._collection.get(limit=1, include=["embeddings"])["embeddings"]
        return len(sample[0]) if sample is not None and len(sample) else None
