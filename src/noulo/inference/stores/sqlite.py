"""SQLite vector store: stdlib ``sqlite3`` plus brute-force cosine in numpy.

Suited to local, single-node use (thousands to low hundreds of thousands of
records per task). Embeddings are stored as float32 BLOBs.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

from noulo.inference.stores import MemoryRecord

_COLUMNS = (
    "id",
    "primitive",
    "task",
    "input",
    "input_norm",
    "model_id",
    "embedder_id",
    "embedding",
    "observed_value",
    "observed_choice",
    "verified_value",
    "verified_choice",
    "hits",
    "created_at",
    "updated_at",
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM records"
_UPSERT = (
    f"INSERT INTO records ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)})"
    " ON CONFLICT(id) DO UPDATE SET "
    + ", ".join(f"{c} = excluded.{c}" for c in _COLUMNS if c != "id")
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id              TEXT PRIMARY KEY,
    primitive       TEXT NOT NULL,
    task            TEXT NOT NULL,
    input           TEXT NOT NULL,
    input_norm      TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    embedder_id     TEXT NOT NULL,
    embedding       BLOB NOT NULL,
    observed_value  REAL,
    observed_choice TEXT,
    verified_value  REAL,
    verified_choice TEXT,
    hits            INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS records_search ON records (primitive, task, embedder_id);
CREATE INDEX IF NOT EXISTS records_dedup ON records (primitive, task, input_norm, model_id);
CREATE INDEX IF NOT EXISTS records_created ON records (created_at);
"""


def _to_row(record: MemoryRecord) -> tuple:
    values = {c: getattr(record, c) for c in _COLUMNS}
    values["embedding"] = np.asarray(record.embedding, dtype=np.float32).tobytes()
    return tuple(values[c] for c in _COLUMNS)


def _from_row(row: tuple) -> MemoryRecord:
    values = dict(zip(_COLUMNS, row, strict=True))
    values["embedding"] = np.frombuffer(values["embedding"], dtype=np.float32).copy()
    return MemoryRecord(**values)


class SqliteVectorStore:
    """Vector store in a single SQLite file (WAL mode) or ``":memory:"``."""

    def __init__(
        self,
        *,
        location: str | None = None,
        collection: str = "noulo_memory",
        api_key: str | None = None,
        dim: int | None = None,
    ) -> None:
        # ``collection``/``api_key``/``dim`` are accepted for factory symmetry; one
        # SQLite file is one collection and vector widths are free-form.
        path = location or ":memory:"
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(path, check_same_thread=False)
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def upsert(self, record: MemoryRecord) -> None:
        with self._lock, self._db() as conn:
            conn.execute(_UPSERT, _to_row(record))

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._fetch_one(f"{_SELECT} WHERE id = ?", (record_id,))

    def find_duplicate(
        self, *, primitive: str, task: str, input_norm: str, model_id: str
    ) -> MemoryRecord | None:
        return self._fetch_one(
            f"{_SELECT} WHERE primitive = ? AND task = ? AND input_norm = ? AND model_id = ?"
            " LIMIT 1",
            (primitive, task, input_norm, model_id),
        )

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
        candidates = [
            record
            for record in self._fetch_all(
                f"{_SELECT} WHERE primitive = ? AND task = ? AND embedder_id = ?",
                (primitive, task, embedder_id),
            )
            if record.embedding.shape == query.shape
        ]
        if not candidates or top_k <= 0:
            return []
        matrix = np.stack([record.embedding for record in candidates])
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        sims = (matrix @ query) / np.maximum(norms, 1e-12)
        order = np.argsort(-sims, kind="stable")[:top_k]
        return [(candidates[i], float(sims[i])) for i in order if sims[i] >= min_similarity]

    def list(self, *, limit: int, offset: int, primitive: str | None = None) -> list[MemoryRecord]:
        where, params = ("WHERE primitive = ?", (primitive,)) if primitive else ("", ())
        return self._fetch_all(
            f"{_SELECT} {where} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )

    def count(self) -> dict[str, int]:
        with self._lock:
            total, verified = (
                self._db()
                .execute(
                    "SELECT COUNT(*), COUNT(*) FILTER (WHERE verified_value IS NOT NULL"
                    " OR verified_choice IS NOT NULL) FROM records"
                )
                .fetchone()
            )
        return {"records": total, "verified": verified}

    def clear(self) -> int:
        with self._lock, self._db() as conn:
            return conn.execute("DELETE FROM records").rowcount

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Vector store is closed.")
        return self._conn

    def _fetch_one(self, sql: str, params: tuple) -> MemoryRecord | None:
        with self._lock:
            row = self._db().execute(sql, params).fetchone()
        return _from_row(row) if row is not None else None

    def _fetch_all(self, sql: str, params: tuple) -> list[MemoryRecord]:
        with self._lock:
            rows = self._db().execute(sql, params).fetchall()
        return [_from_row(row) for row in rows]
