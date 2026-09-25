import threading
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from openapi_spec_validator import validate

from noulo import __version__
from noulo.api.server import create_app
from noulo.config import Settings
from noulo.inference.engine import DecisionEngine
from noulo.inference.types import BackendError, ModelLoadError
from noulo.registry import ModelRegistry
from tests.fakes import FakeBackend, FakeLoader

NOUL = {
    "input": "The invoice has remained unpaid for 120 days.",
    "proposition": "The customer has an overdue payment.",
}
CHOICE = {
    "input": "Payment taken twice.",
    "question": "Which department should handle this?",
    "choices": [
        {"id": "A", "text": "Billing"},
        {"id": "B", "text": "Technical Support"},
        {"id": "C", "text": "Sales"},
    ],
}
SCORE = {
    "input": "Production is down for everyone.",
    "question": "How severe is this incident?",
    "rubric": ["insignificant", "low", "medium", "high", "critical"],
}


def settings(**overrides) -> Settings:
    base = {"ui_enabled": False, "learning_enabled": False, "models_file": None}
    return Settings(_env_file=None, **{**base, **overrides})


def build(backends=None, *, cfg=None, tmp_path=None, **engine_kwargs):
    loader = FakeLoader(**(backends or {"fake-model": FakeBackend(choice_probs=[0.8, 0.1, 0.1])}))
    first = next(iter(loader.backends))
    engine = DecisionEngine(load_backend=loader, model_id=first, **engine_kwargs)
    cfg = cfg or settings()
    registry = ModelRegistry(tmp_path or "models-none", None)
    return create_app(cfg, engine=engine, registry=registry), engine, loader


@contextmanager
def running(**kwargs):
    app, engine, loader = build(**kwargs)
    with TestClient(app) as client:
        client.engine, client.loader = engine, loader
        yield client


# ------------------------------------------------------------------ health / info


def test_health_reports_ok_after_model_loaded():
    with running() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "modelLoaded": True}


def test_health_is_not_ready_before_startup_completes():
    app, _, _ = build()
    client = TestClient(app)  # lifespan not started: model not loaded yet
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["modelLoaded"] is False


def test_model_is_loaded_once_at_startup_not_per_request():
    with running() as client:
        for _ in range(3):
            client.post("/api/v1/noul", json=NOUL)
        assert client.loader.loads == ["fake-model"]


def test_startup_fails_when_model_cannot_load():
    app, _, _ = build({"fake-model": FakeBackend(warmup_error=ModelLoadError("broken"))})
    with pytest.raises(ModelLoadError), TestClient(app):
        pass


def test_info_matches_spec_shape_without_paths():
    with running() as client:
        body = client.get("/api/v1/info").json()
    assert body["name"] == "noulo"
    assert body["version"] == __version__
    assert body["model"] == "fake-model"
    assert body["quantization"] == "INT8"
    assert body["capabilities"] == ["choice", "score", "noul"]
    assert "/" not in str(body["model"])


# ------------------------------------------------------------------ primitives


def test_noul_endpoint_returns_strict_contract():
    with running() as client:
        response = client.post("/api/v1/noul", json=NOUL)
    assert response.status_code == 200
    assert response.json() == {"type": "noul", "value": pytest.approx(0.8)}


def test_choice_endpoint_returns_a_supplied_id():
    with running() as client:
        body = client.post("/api/v1/choice", json=CHOICE).json()
    assert body == {"type": "choice", "value": "A"}


def test_score_endpoint_returns_value_in_unit_interval():
    with running() as client:
        body = client.post("/api/v1/score", json=SCORE).json()
    assert body["type"] == "score" and 0.0 <= body["value"] <= 1.0


@pytest.mark.parametrize(
    "payload,kind",
    [
        ({"type": "noul", **NOUL}, "noul"),
        ({"type": "choice", **CHOICE}, "choice"),
        ({"type": "score", **SCORE}, "score"),
    ],
)
def test_evaluate_routes_to_same_primitive(payload, kind):
    with running() as client:
        unified = client.post("/api/v1/evaluate", json=payload).json()
        dedicated = client.post(f"/api/v1/{kind}", json=payload).json()
    assert unified == dedicated and unified["type"] == kind


def test_diagnostics_are_opt_in():
    with running() as client:
        body = client.post("/api/v1/choice?diagnostics=true", json=CHOICE).json()
    assert body["value"] == "A"
    assert body["diagnostics"]["probabilities"]["A"] == pytest.approx(0.8)


# ------------------------------------------------------------------ validation / errors


def test_missing_proposition_returns_spec_error():
    with running() as client:
        response = client.post("/api/v1/noul", json={"input": "x"})
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "INVALID_REQUEST", "message": "Noul requires a proposition."}
    }


def test_malformed_json_is_400():
    with running() as client:
        response = client.post(
            "/api/v1/noul", content=b"{not json", headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MALFORMED_JSON"


def test_unsupported_primitive_is_rejected():
    with running() as client:
        response = client.post("/api/v1/evaluate", json={"type": "vibes", **NOUL})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_PRIMITIVE"


def test_duplicate_choice_ids_are_rejected():
    payload = {**CHOICE, "choices": [{"id": "A", "text": "x"}, {"id": "A", "text": "y"}]}
    with running() as client:
        assert client.post("/api/v1/choice", json=payload).status_code == 422


def test_oversized_body_is_413():
    with running(cfg=settings(max_body_bytes=100)) as client:
        response = client.post("/api/v1/noul", json={**NOUL, "input": "x" * 500})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_input_over_character_limit_is_413():
    with running(cfg=settings(max_input_chars=10)) as client:
        response = client.post("/api/v1/noul", json=NOUL)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "INPUT_TOO_LARGE"


def test_unknown_route_uses_error_shape():
    with running() as client:
        response = client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_local_inference_failure_is_500_without_internals():
    with running(
        backends={"fake-model": FakeBackend(error=RuntimeError("secret stack detail"))}
    ) as client:
        response = client.post("/api/v1/noul", json=NOUL)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INFERENCE_ERROR"
    assert "secret stack detail" not in response.text


def test_remote_backend_failure_is_502():
    with running(
        backends={"remote": FakeBackend("remote", local=False, error=BackendError("x"))}
    ) as client:
        response = client.post("/api/v1/noul", json=NOUL)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "UPSTREAM_ERROR"


def test_busy_engine_returns_503_with_retry_after():
    gate = threading.Event()
    backend = FakeBackend(gate=gate)
    with running(backends={"fake-model": backend}, max_concurrency=1, max_queue=0) as client:
        worker = threading.Thread(target=client.post, args=("/api/v1/noul",), kwargs={"json": NOUL})
        worker.start()
        assert backend.started.wait(2)
        response = client.post("/api/v1/noul", json=NOUL)
        gate.set()
        worker.join()
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ENGINE_BUSY"
    assert response.headers["Retry-After"] == "1"


# ------------------------------------------------------------------ concurrency / shutdown


def test_parallel_requests_respect_concurrency_limit():
    backend = FakeBackend(delay=0.01)
    with running(backends={"fake-model": backend}, max_concurrency=2, max_queue=64) as client:
        results = []

        def call():
            results.append(client.post("/api/v1/noul", json=NOUL).status_code)

        threads = [threading.Thread(target=call) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert results == [200] * 12
    assert backend.max_active <= 2


def test_shutdown_releases_model():
    backend = FakeBackend()
    with running(backends={"fake-model": backend}) as client:
        client.post("/api/v1/noul", json=NOUL)
    assert backend.closed
    assert client.engine.state == "stopped"


# ------------------------------------------------------------------ security


def test_api_key_required_when_configured():
    with running(cfg=settings(api_key="local-key")) as client:
        assert client.post("/api/v1/noul", json=NOUL).status_code == 401
        wrong = client.post("/api/v1/noul", json=NOUL, headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        assert wrong.json()["error"]["code"] == "UNAUTHORIZED"
        ok = client.post("/api/v1/noul", json=NOUL, headers={"Authorization": "Bearer local-key"})
        assert ok.status_code == 200
        assert client.get("/health").status_code == 200  # health stays open


def test_cors_allows_localhost_origins_by_default():
    with running() as client:
        response = client.options(
            "/api/v1/noul",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_unlisted_origins():
    with running() as client:
        response = client.options(
            "/api/v1/noul",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_configured_origins():
    with running(cfg=settings(cors_origins="https://app.example")) as client:
        response = client.post("/api/v1/noul", json=NOUL, headers={"Origin": "https://app.example"})
    assert response.headers.get("access-control-allow-origin") == "https://app.example"


def test_cors_can_be_disabled():
    with running(cfg=settings(cors_enabled=False)) as client:
        response = client.post(
            "/api/v1/noul", json=NOUL, headers={"Origin": "http://localhost:5173"}
        )
    assert "access-control-allow-origin" not in response.headers


# ------------------------------------------------------------------ models


def test_models_endpoint_lists_models_and_active_one(tmp_path):
    with running(tmp_path=tmp_path) as client:
        body = client.get("/api/v1/models").json()
    assert body["active"] == "fake-model"
    assert any(m["id"] == "nli-deberta-v3-xsmall-int8" for m in body["models"])


def test_switch_active_model_at_runtime():
    backends = {"m1": FakeBackend("m1", noul_value=0.9), "m2": FakeBackend("m2", noul_value=0.1)}
    with running(backends=backends) as client:
        response = client.put("/api/v1/models/active", json={"id": "m2"})
        assert response.status_code == 200
        assert response.json()["active"]["id"] == "m2"
        assert client.get("/api/v1/info").json()["model"] == "m2"
        assert client.post("/api/v1/noul", json=NOUL).json()["value"] == pytest.approx(0.1)


def test_switch_to_unknown_model_is_404_and_keeps_current():
    with running() as client:
        response = client.put("/api/v1/models/active", json={"id": "ghost"})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"
        assert client.get("/api/v1/info").json()["model"] == "fake-model"


# ------------------------------------------------------------------ openapi


def test_openapi_document_is_valid_and_complete():
    with running() as client:
        spec = client.get("/openapi.json").json()
    validate(spec)
    for path in [
        "/health",
        "/api/v1/info",
        "/api/v1/evaluate",
        "/api/v1/choice",
        "/api/v1/score",
        "/api/v1/noul",
        "/api/v1/models",
        "/api/v1/models/active",
    ]:
        assert path in spec["paths"], path
    schemas = spec["components"]["schemas"]
    for name in ["NoulRequest", "ChoiceRequest", "ScoreRequest", "ErrorResponse"]:
        assert name in schemas, name
    noul_body = spec["paths"]["/api/v1/noul"]["post"]["requestBody"]["content"]["application/json"]
    assert noul_body["schema"] == {"$ref": "#/components/schemas/NoulRequest"}
