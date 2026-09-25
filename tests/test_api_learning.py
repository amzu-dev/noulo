from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from noulo.api.server import create_app
from noulo.config import Settings
from noulo.inference.embedder import HashingEmbedder
from noulo.inference.engine import DecisionEngine
from noulo.inference.memory import LearningMemory
from noulo.inference.stores import open_store
from noulo.registry import ModelRegistry
from tests.fakes import FakeBackend, FakeLoader

NOUL = {"input": "I was charged twice this month.", "proposition": "The customer is happy."}
CHOICE = {
    "input": "I was charged twice this month.",
    "question": "Which team?",
    "choices": [{"id": "A", "text": "Billing"}, {"id": "B", "text": "Sales"}],
}


@contextmanager
def running(learning=True, backend=None, tmp_path="models-none"):
    engine = DecisionEngine(
        load_backend=FakeLoader(
            **{"m": backend or FakeBackend("m", noul_value=0.9, choice_probs=[0.8, 0.2])}
        ),
        model_id="m",
        open_memory=lambda: LearningMemory(
            open_store("sqlite", location=":memory:"), HashingEmbedder()
        ),
        learning=learning,
    )
    cfg = Settings(_env_file=None, ui_enabled=False, models_file=None)
    with TestClient(
        create_app(cfg, engine=engine, registry=ModelRegistry(tmp_path, None))
    ) as client:
        yield client


def test_learning_status_reports_enabled_and_stats():
    with running() as client:
        body = client.get("/api/v1/learning").json()
    assert body["enabled"] is True and body["stats"]["records"] == 0


def test_evaluation_returns_record_id_header_but_strict_body():
    with running() as client:
        response = client.post("/api/v1/noul", json=NOUL)
    assert response.headers["X-Record-Id"]
    assert response.json() == {"type": "noul", "value": pytest.approx(0.9)}


def test_feedback_by_record_id_corrects_next_answer():
    with running() as client:
        record_id = client.post("/api/v1/noul", json=NOUL).headers["X-Record-Id"]
        response = client.post("/api/v1/feedback", json={"recordId": record_id, "expected": False})
        assert response.status_code == 200
        assert response.json() == {"recordId": record_id, "verified": True}
        assert client.post("/api/v1/noul", json=NOUL).json()["value"] < 0.5


def test_feedback_with_full_request_teaches_directly():
    with running() as client:
        response = client.post(
            "/api/v1/feedback", json={"request": {"type": "choice", **CHOICE}, "expected": "B"}
        )
        assert response.status_code == 200
        assert client.post("/api/v1/choice", json=CHOICE).json()["value"] == "B"


def test_feedback_with_invalid_request_uses_validation_messages():
    with running() as client:
        response = client.post(
            "/api/v1/feedback", json={"request": {"type": "noul", "input": "x"}, "expected": 1}
        )
    assert response.status_code == 422
    assert response.json()["error"]["message"] == "Noul requires a proposition."


def test_feedback_needs_record_id_or_request():
    with running() as client:
        response = client.post("/api/v1/feedback", json={"expected": 1})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_feedback_for_unknown_record_is_404():
    with running() as client:
        response = client.post("/api/v1/feedback", json={"recordId": "ghost", "expected": 1})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RECORD_NOT_FOUND"


def test_feedback_with_wrong_outcome_kind_is_422():
    with running() as client:
        record_id = client.post("/api/v1/noul", json=NOUL).headers["X-Record-Id"]
        response = client.post("/api/v1/feedback", json={"recordId": record_id, "expected": "A"})
    assert response.status_code == 422


def test_learning_can_be_toggled_off_via_api():
    with running() as client:
        body = client.put("/api/v1/learning", json={"enabled": False}).json()
        assert body["enabled"] is False
        response = client.post("/api/v1/noul", json=NOUL)
        assert "X-Record-Id" not in response.headers
        feedback = client.post("/api/v1/feedback", json={"recordId": "x", "expected": 1})
        assert feedback.status_code == 409
        assert feedback.json()["error"]["code"] == "LEARNING_DISABLED"
        assert client.get("/api/v1/info").json()["learning"] is False


def test_learning_can_be_toggled_on_when_started_disabled():
    with running(learning=False) as client:
        assert client.put("/api/v1/learning", json={"enabled": True}).json()["enabled"] is True
        assert client.post("/api/v1/noul", json=NOUL).headers.get("X-Record-Id")


def test_records_listing_and_filter():
    with running() as client:
        client.post("/api/v1/noul", json=NOUL)
        client.post("/api/v1/choice", json=CHOICE)
        records = client.get("/api/v1/learning/records").json()["records"]
        assert {r["primitive"] for r in records} == {"noul", "choice"}
        only = client.get("/api/v1/learning/records?type=choice&limit=5").json()["records"]
        assert [r["primitive"] for r in only] == ["choice"]


def test_clear_records():
    with running() as client:
        client.post("/api/v1/noul", json=NOUL)
        assert client.delete("/api/v1/learning/records").json() == {"deleted": 1}
        assert client.get("/api/v1/learning").json()["stats"]["records"] == 0


def test_openapi_documents_learning_endpoints():
    with running() as client:
        paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/v1/learning", "/api/v1/learning/records", "/api/v1/feedback"):
        assert path in paths


# ---------------------------------------------------------------- import (teach from a file)


def test_import_teaches_many_examples_at_once():
    items = [
        {"type": "noul", **NOUL, "expected": False},
        {"type": "choice", **CHOICE, "expected": "B"},
    ]
    with running() as client:
        response = client.post("/api/v1/learning/import", json={"items": items})
        assert response.status_code == 200
        assert response.json() == {"imported": 2, "failed": []}
        assert client.post("/api/v1/noul", json=NOUL).json()["value"] < 0.5
        assert client.post("/api/v1/choice", json=CHOICE).json()["value"] == "B"


def test_import_reports_bad_examples_without_dropping_good_ones():
    items = [
        {"type": "noul", **NOUL, "expected": True},
        {"type": "noul", "input": "x", "expected": True},
        {"type": "choice", **CHOICE, "expected": "Z"},
    ]
    with running() as client:
        body = client.post("/api/v1/learning/import", json={"items": items}).json()
    assert body["imported"] == 1
    assert [(f["index"], f["code"]) for f in body["failed"]] == [
        (1, "INVALID_REQUEST"),
        (2, "INVALID_REQUEST"),
    ]
    assert body["failed"][0]["message"] == "Noul requires a proposition."
    assert "supplied" in body["failed"][1]["message"]


def test_import_accepts_benchmark_dataset_rows():
    row = {
        "id": "noul-1",
        "input": NOUL["input"],
        "proposition": NOUL["proposition"],
        "label": "no",
        "split": "calibration",
    }
    with running() as client:
        assert client.post("/api/v1/learning/import", json={"items": [row]}).json()["imported"] == 1


def test_import_needs_learning_on():
    with running(learning=False) as client:
        response = client.post("/api/v1/learning/import", json={"items": []})
    assert response.status_code == 409 and response.json()["error"]["code"] == "LEARNING_DISABLED"


def test_import_rejects_oversized_batches():
    with running() as client:
        response = client.post(
            "/api/v1/learning/import",
            json={"items": [{"type": "noul", **NOUL, "expected": 1}] * 1001},
        )
    assert response.status_code == 422


def test_info_exposes_request_limits_for_clients():
    with running() as client:
        limits = client.get("/api/v1/info").json()["limits"]
    assert limits["maxBodyBytes"] == 65536 and limits["maxImportItems"] == 1000
