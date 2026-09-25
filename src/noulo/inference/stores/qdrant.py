"""Qdrant vector store: embedded (local directory / ``":memory:"``) or remote server.

Every non-vector field lives in the point payload. Record ids are arbitrary
strings, so point ids are derived deterministically with ``uuid5``. The
collection uses cosine distance and is created on the first upsert (or eagerly
when ``dim`` is given), because its vector size is fixed at creation time.

Embedded Qdrant persists through ``sqlite3`` connections bound to the thread
that opened them, so in embedded mode every client call runs on one dedicated
thread (:class:`_PinnedClient`); the store is then safe to use from any thread.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from noulo.inference.stores import MemoryRecord

try:
    from qdrant_client import QdrantClient, models
except ImportError as exc:
    raise ImportError(
        "The 'qdrant' vector store needs qdrant-client: pip install 'noulo[qdrant]'"
    ) from exc

_POINT_NAMESPACE = uuid.UUID("6f1d7c2e-3b0a-4c55-9d6e-8a4f0b7e2c91")
_PAGE = 256
_KEYWORD_FIELDS = ("primitive", "task", "embedder_id", "input_norm", "model_id")


def _point_id(record_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, record_id))


def _payload(record: MemoryRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "primitive": record.primitive,
        "task": record.task,
        "input": record.input,
        "input_norm": record.input_norm,
        "model_id": record.model_id,
        "embedder_id": record.embedder_id,
        "observed_value": record.observed_value,
        "observed_choice": record.observed_choice,
        "verified_value": record.verified_value,
        "verified_choice": record.verified_choice,
        "verified": record.verified,
        "hits": record.hits,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _record(point: Any) -> MemoryRecord:
    payload = dict(point.payload)
    payload.pop("verified", None)
    return MemoryRecord(embedding=np.asarray(point.vector, dtype=np.float32), **payload)


def _match(**conditions: Any) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in conditions.items()
        ]
    )


class _PinnedClient:
    """Proxy that runs every client method on a single dedicated thread."""

    def __init__(self, factory: Callable[[], QdrantClient]) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qdrant-embedded")
        self._client = self._executor.submit(factory).result()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if not callable(attribute):
            return attribute
        return lambda *args, **kwargs: self._executor.submit(attribute, *args, **kwargs).result()

    def close(self) -> None:
        client, self._client = self._client, None

        def close_and_release() -> None:
            # QdrantClient.__del__ closes again; drop the last reference here so
            # that also happens on the owning thread.
            nonlocal client
            client.close()
            client = None

        try:
            self._executor.submit(close_and_release).result()
        finally:
            self._executor.shutdown(wait=True)


class QdrantVectorStore:
    """Vector store backed by a Qdrant collection."""

    def __init__(
        self,
        *,
        location: str | None = None,
        collection: str = "noulo_memory",
        api_key: str | None = None,
        dim: int | None = None,
    ) -> None:
        location = location or ":memory:"
        self._remote = location.startswith(("http://", "https://"))
        self._client: Any
        if self._remote:
            self._client = QdrantClient(url=location, api_key=api_key)
        elif location == ":memory:":
            self._client = _PinnedClient(lambda: QdrantClient(location=":memory:"))
        else:
            Path(location).mkdir(parents=True, exist_ok=True)
            self._client = _PinnedClient(lambda: QdrantClient(path=location))
        self._collection = collection
        self._closed = False
        self._dim: int | None = self._existing_dim()
        if self._dim is None and dim:
            self._create(dim)

    def upsert(self, record: MemoryRecord) -> None:
        vector = np.asarray(record.embedding, dtype=np.float32)
        if self._dim is None:
            self._create(len(vector))
        elif len(vector) != self._dim:
            raise ValueError(
                f"Collection '{self._collection}' holds {self._dim}-d vectors, got {len(vector)}; "
                "use a separate collection per embedder width."
            )
        self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=_point_id(record.id), vector=vector.tolist(), payload=_payload(record)
                )
            ],
            wait=True,
        )

    def get(self, record_id: str) -> MemoryRecord | None:
        if self._dim is None:
            return None
        points = self._client.retrieve(
            self._collection, ids=[_point_id(record_id)], with_payload=True, with_vectors=True
        )
        return _record(points[0]) if points else None

    def find_duplicate(
        self, *, primitive: str, task: str, input_norm: str, model_id: str
    ) -> MemoryRecord | None:
        if self._dim is None:
            return None
        points, _ = self._client.scroll(
            self._collection,
            scroll_filter=_match(
                primitive=primitive, task=task, input_norm=input_norm, model_id=model_id
            ),
            limit=1,
            with_payload=True,
            with_vectors=True,
        )
        return _record(points[0]) if points else None

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
        response = self._client.query_points(
            self._collection,
            query=query.tolist(),
            query_filter=_match(primitive=primitive, task=task, embedder_id=embedder_id),
            limit=top_k,
            score_threshold=min_similarity,
            with_payload=True,
            with_vectors=True,
        )
        return [(_record(point), float(point.score)) for point in response.points]

    def list(self, *, limit: int, offset: int, primitive: str | None = None) -> list[MemoryRecord]:
        # Qdrant's order_by needs a payload index and does not support offsets,
        # so we scroll the (filtered) collection and sort client-side.
        records = self._scroll_all(_match(primitive=primitive) if primitive else None)
        records.sort(key=lambda r: (r.created_at, r.id), reverse=True)
        return records[offset : offset + limit]

    def count(self) -> dict[str, int]:
        if self._dim is None:
            return {"records": 0, "verified": 0}
        total = self._client.count(self._collection, exact=True).count
        verified = self._client.count(
            self._collection, count_filter=_match(verified=True), exact=True
        ).count
        return {"records": total, "verified": verified}

    def clear(self) -> int:
        removed = self.count()["records"]
        if self._dim is not None:
            self._client.delete_collection(self._collection)
            self._dim = None
        return removed

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.close()

    def _existing_dim(self) -> int | None:
        if not self._client.collection_exists(self._collection):
            return None
        vectors = self._client.get_collection(self._collection).config.params.vectors
        return int(vectors.size)

    def _create(self, dim: int) -> None:
        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        if self._remote:  # payload indexes are a no-op (with a warning) in embedded mode
            for field in _KEYWORD_FIELDS:
                self._client.create_payload_index(
                    self._collection, field, models.PayloadSchemaType.KEYWORD
                )
            self._client.create_payload_index(
                self._collection, "verified", models.PayloadSchemaType.BOOL
            )
        self._dim = dim

    def _scroll_all(self, scroll_filter: models.Filter | None) -> list[MemoryRecord]:
        if self._dim is None:
            return []
        records: list[MemoryRecord] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._collection,
                scroll_filter=scroll_filter,
                limit=_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            records.extend(_record(point) for point in points)
            if offset is None:
                return records
