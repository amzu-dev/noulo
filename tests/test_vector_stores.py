"""Contract tests every VectorStore implementation must pass."""

from __future__ import annotations

import os
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from noulo.inference.stores import MemoryRecord, VectorStore, open_store


@dataclass(frozen=True)
class StoreCase:
    kind: str
    location: Callable[[Path], str | None]
    persistent: bool


def _remote(env: str, kind: str) -> pytest.param:
    url = os.environ.get(env)
    return pytest.param(
        StoreCase(kind, lambda _tmp: url, persistent=True),
        id=f"{kind}-remote",
        marks=pytest.mark.skipif(not url, reason=f"set {env} to test a remote {kind} server"),
    )


STORE_CASES = [
    pytest.param(StoreCase("sqlite", lambda _tmp: ":memory:", False), id="sqlite-memory"),
    pytest.param(StoreCase("sqlite", lambda tmp: str(tmp / "mem.sqlite3"), True), id="sqlite-file"),
    pytest.param(StoreCase("qdrant", lambda _tmp: ":memory:", False), id="qdrant-memory"),
    pytest.param(StoreCase("qdrant", lambda tmp: str(tmp / "qdrant"), True), id="qdrant-local"),
    _remote("NOULO_TEST_QDRANT_URL", "qdrant"),
    pytest.param(StoreCase("chroma", lambda _tmp: None, False), id="chroma-ephemeral"),
    pytest.param(
        StoreCase("chroma", lambda tmp: str(tmp / "chroma"), True), id="chroma-persistent"
    ),
    _remote("NOULO_TEST_CHROMA_URL", "chroma"),
]


@pytest.fixture(params=STORE_CASES)
def case(request) -> StoreCase:
    return request.param


@pytest.fixture
def open_case(case: StoreCase, tmp_path: Path):
    """Callable that (re)opens the store under test at the same location."""
    collection = f"test_{uuid.uuid4().hex[:12]}"
    opened: list[VectorStore] = []

    def _open() -> VectorStore:
        store = open_store(case.kind, location=case.location(tmp_path), collection=collection)
        opened.append(store)
        return store

    yield _open
    for store in opened:
        if case.location(tmp_path) and case.location(tmp_path).startswith("http"):
            store.clear()
        store.close()


@pytest.fixture
def store(open_case) -> VectorStore:
    return open_case()


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _record(
    record_id: str,
    *,
    vector: np.ndarray | None = None,
    primitive: str = "noul",
    task: str = "noul|the invoice is overdue",
    input: str = "Invoice 4411 is unpaid.",
    model_id: str = "m1",
    embedder_id: str = "hashing-4",
    created_at: str = "2026-01-01T00:00:00.000000Z",
    **fields,
) -> MemoryRecord:
    defaults = {
        "observed_value": 0.9 if primitive != "choice" else None,
        "observed_choice": "billing" if primitive == "choice" else None,
        "verified_value": None,
        "verified_choice": None,
        "hits": 1,
        "updated_at": created_at,
    }
    defaults.update(fields)
    return MemoryRecord(
        id=record_id,
        primitive=primitive,
        task=task,
        input=input,
        input_norm=" ".join(input.lower().split()),
        model_id=model_id,
        embedder_id=embedder_id,
        embedding=vector if vector is not None else _unit(1, 0, 0, 0),
        created_at=created_at,
        **defaults,
    )


def _stamp(second: int) -> str:
    return f"2026-01-01T00:00:{second:02d}.000000Z"


def test_store_satisfies_protocol(store):
    assert isinstance(store, VectorStore)


def test_empty_store_answers_every_query(store):
    assert store.get("nope") is None
    assert store.find_duplicate(primitive="noul", task="t", input_norm="x", model_id="m") is None
    assert (
        store.search(
            primitive="noul",
            task="t",
            embedder_id="e",
            vector=_unit(1, 0, 0, 0),
            top_k=5,
            min_similarity=0.0,
        )
        == []
    )
    assert store.list(limit=10, offset=0) == []
    assert store.count() == {"records": 0, "verified": 0}
    assert store.clear() == 0


def test_upsert_then_get_round_trips_every_field(store):
    original = _record(
        "rec-1",
        vector=_unit(0.2, 0.4, 0.1, 0.9),
        input="Invoice  4411 is UNPAID.",
        verified_value=0.0,
        hits=3,
        updated_at=_stamp(9),
    )
    store.upsert(original)
    loaded = store.get("rec-1")
    assert loaded is not None
    assert loaded.embedding.dtype == np.float32
    np.testing.assert_allclose(loaded.embedding, original.embedding, rtol=1e-5, atol=1e-6)
    assert replace(loaded, embedding=None) == replace(original, embedding=None)


def test_round_trips_choice_outcomes_and_nulls(store):
    original = _record("choice-1", primitive="choice", task="choice|q|a=x", verified_choice="it")
    store.upsert(original)
    loaded = store.get("choice-1")
    assert loaded is not None
    assert (loaded.observed_value, loaded.observed_choice) == (None, "billing")
    assert (loaded.verified_value, loaded.verified_choice) == (None, "it")


def test_upsert_existing_id_replaces_record(store):
    store.upsert(_record("rec-1", observed_value=0.9))
    store.upsert(_record("rec-1", observed_value=0.2, hits=2, updated_at=_stamp(5)))
    loaded = store.get("rec-1")
    assert loaded is not None
    assert loaded.observed_value == pytest.approx(0.2)
    assert loaded.hits == 2
    assert loaded.updated_at == _stamp(5)
    assert store.count()["records"] == 1


def test_find_duplicate_matches_primitive_task_input_and_model(store):
    store.upsert(_record("rec-1"))
    found = store.find_duplicate(
        primitive="noul",
        task="noul|the invoice is overdue",
        input_norm="invoice 4411 is unpaid.",
        model_id="m1",
    )
    assert found is not None and found.id == "rec-1"
    for override in (
        {"primitive": "score"},
        {"task": "noul|other"},
        {"input_norm": "invoice 4412 is unpaid."},
        {"model_id": "m2"},
    ):
        query = {
            "primitive": "noul",
            "task": "noul|the invoice is overdue",
            "input_norm": "invoice 4411 is unpaid.",
            "model_id": "m1",
            **override,
        }
        assert store.find_duplicate(**query) is None, override


def _search(store: VectorStore, vector: np.ndarray, **overrides):
    query = {
        "primitive": "noul",
        "task": "noul|the invoice is overdue",
        "embedder_id": "hashing-4",
        "vector": vector,
        "top_k": 10,
        "min_similarity": 0.0,
        **overrides,
    }
    return store.search(**query)


def test_search_filters_by_primitive_task_and_embedder(store):
    vector = _unit(1, 0, 0, 0)
    store.upsert(_record("match", vector=vector))
    store.upsert(_record("other-primitive", vector=vector, primitive="score"))
    store.upsert(_record("other-task", vector=vector, task="noul|different"))
    store.upsert(_record("other-embedder", vector=vector, embedder_id="minilm"))
    results = _search(store, vector)
    assert [record.id for record, _ in results] == ["match"]
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_search_orders_by_cosine_and_respects_top_k(store):
    query = _unit(1, 0, 0, 0)
    store.upsert(_record("far", vector=_unit(1, 1, 1, 0)))
    store.upsert(_record("near", vector=_unit(1, 0.1, 0, 0)))
    store.upsert(_record("mid", vector=_unit(1, 0.6, 0, 0)))
    results = _search(store, query, top_k=2)
    assert [record.id for record, _ in results] == ["near", "mid"]
    expected = [float(_unit(1, 0.1, 0, 0) @ query), float(_unit(1, 0.6, 0, 0) @ query)]
    assert [score for _, score in results] == pytest.approx(expected, abs=1e-4)
    np.testing.assert_allclose(results[0][0].embedding, _unit(1, 0.1, 0, 0), atol=1e-5)


def test_search_with_other_vector_width_finds_nothing(store):
    store.upsert(_record("four-d", vector=_unit(1, 0, 0, 0)))
    assert _search(store, _unit(1, 0, 0, 0, 0, 0)) == []


def test_search_drops_results_below_min_similarity(store):
    query = _unit(1, 0, 0, 0)
    store.upsert(_record("near", vector=_unit(1, 0.1, 0, 0)))
    store.upsert(_record("orthogonal", vector=_unit(0, 1, 0, 0)))
    results = _search(store, query, min_similarity=0.8)
    assert [record.id for record, _ in results] == ["near"]


def test_list_is_newest_first_with_paging_and_primitive_filter(store):
    for second, (record_id, primitive) in enumerate(
        [("a", "noul"), ("b", "choice"), ("c", "noul"), ("d", "noul"), ("e", "score")]
    ):
        task = "choice|q|x=y" if primitive == "choice" else "t"
        store.upsert(_record(record_id, primitive=primitive, task=task, created_at=_stamp(second)))
    assert [r.id for r in store.list(limit=10, offset=0)] == ["e", "d", "c", "b", "a"]
    assert [r.id for r in store.list(limit=2, offset=1)] == ["d", "c"]
    assert [r.id for r in store.list(limit=10, offset=0, primitive="noul")] == ["d", "c", "a"]
    assert [r.id for r in store.list(limit=1, offset=1, primitive="noul")] == ["c"]


def test_count_reports_records_and_verified(store):
    store.upsert(_record("plain"))
    store.upsert(_record("value", verified_value=0.0))
    store.upsert(_record("choice", primitive="choice", task="c", verified_choice="it"))
    assert store.count() == {"records": 3, "verified": 2}


def test_clear_removes_everything_and_store_stays_usable(store):
    store.upsert(_record("a"))
    store.upsert(_record("b"))
    assert store.clear() == 2
    assert store.count() == {"records": 0, "verified": 0}
    assert store.get("a") is None
    store.upsert(_record("c"))
    assert store.count()["records"] == 1


def test_store_is_usable_across_threads(open_case):
    store = open_case()
    failures: list[BaseException] = []

    def in_worker(action: Callable[[], object]) -> None:
        def run() -> None:
            try:
                action()
            except BaseException as exc:  # noqa: BLE001 - reported below
                failures.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()

    in_worker(lambda: store.upsert(_record("from-worker")))
    in_worker(lambda: store.upsert(_record("second-worker", input="other")))
    assert failures == []
    assert store.get("from-worker") is not None
    assert store.count()["records"] == 2
    store.close()


def test_close_is_idempotent(open_case):
    store = open_case()
    store.close()
    store.close()


def test_file_backed_stores_persist_across_reopen(case, open_case):
    if not case.persistent:
        pytest.skip("in-memory store")
    first = open_case()
    first.upsert(_record("kept", vector=_unit(0, 1, 0, 0), verified_value=0.25))
    first.close()
    second = open_case()
    loaded = second.get("kept")
    assert loaded is not None and loaded.verified_value == pytest.approx(0.25)
    assert [r.id for r, _ in _search(second, _unit(0, 1, 0, 0))] == ["kept"]


# --- factory ---------------------------------------------------------------


class DictStore:
    """Tiny custom store used to check ``module:Class`` loading."""

    def __init__(self, *, location, collection, api_key=None, dim=None):
        self.options = {"location": location, "collection": collection, "api_key": api_key}
        self.records: dict[str, MemoryRecord] = {}

    def upsert(self, record):
        self.records[record.id] = record

    def get(self, record_id):
        return self.records.get(record_id)

    def find_duplicate(self, *, primitive, task, input_norm, model_id):
        return None

    def search(self, *, primitive, task, embedder_id, vector, top_k, min_similarity):
        return []

    def list(self, *, limit, offset, primitive=None):
        return []

    def count(self):
        return {"records": len(self.records), "verified": 0}

    def clear(self):
        return 0

    def close(self):
        pass


def test_open_store_loads_custom_class_by_import_path():
    store = open_store(f"{__name__}:DictStore", location="somewhere", collection="c1", api_key="k")
    assert isinstance(store, DictStore)
    assert isinstance(store, VectorStore)
    assert store.options == {"location": "somewhere", "collection": "c1", "api_key": "k"}


@pytest.mark.parametrize(
    "kind", ["redis", "", "tests.test_vector_stores:Missing", "no_such_module_xyz:Store"]
)
def test_open_store_rejects_unknown_kinds(kind):
    with pytest.raises(ValueError):
        open_store(kind, location=None)


@pytest.mark.parametrize(("kind", "package"), [("qdrant", "qdrant_client"), ("chroma", "chromadb")])
def test_optional_backends_explain_how_to_install(monkeypatch, kind, package):
    monkeypatch.setitem(sys.modules, package, None)  # simulate "not installed"
    monkeypatch.delitem(sys.modules, f"noulo.inference.stores.{kind}", raising=False)
    with pytest.raises(ImportError, match=rf"pip install 'noulo\[{kind}\]'"):
        open_store(kind, location=None)
