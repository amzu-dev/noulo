import asyncio

import pytest
from fastapi.testclient import TestClient

import noulo
from noulo.api.server import create_app
from noulo.config import Settings
from noulo.embedded import Noulo
from noulo.inference.engine import DecisionEngine
from noulo.registry import ModelRegistry
from tests.fakes import FakeBackend, FakeLoader

NOUL = {
    "type": "noul",
    "input": "The invoice has remained unpaid for 120 days.",
    "proposition": "The customer has an overdue payment.",
}
CHOICE = {
    "type": "choice",
    "input": "Charged twice.",
    "question": "Which department?",
    "choices": [{"id": "A", "text": "Billing"}, {"id": "B", "text": "Sales"}],
}


def fake_engine(**backend_kwargs):
    backend = FakeBackend(
        noul_value=0.96, choice_probs=[0.9, 0.1], score_probs=[0, 1], **backend_kwargs
    )
    return DecisionEngine(load_backend=FakeLoader(**{"fake-model": backend}), model_id="fake-model")


def test_evaluate_returns_strict_contract():
    with Noulo(engine=fake_engine()) as engine:
        assert engine.evaluate(NOUL) == {"type": "noul", "value": pytest.approx(0.96)}


def test_convenience_methods():
    with Noulo(engine=fake_engine()) as engine:
        assert engine.noul("x", "y") == pytest.approx(0.96)
        assert engine.choice("x", "q?", {"A": "Billing", "B": "Sales"}) == "A"
        assert engine.choice("x", "q?", [{"id": "A", "text": "a"}, {"id": "B", "text": "b"}]) == "A"
        assert engine.score("x", "q?", ["low", "high"]) == pytest.approx(1.0)


def test_invalid_requests_raise_the_same_errors_as_http():
    with Noulo(engine=fake_engine()) as engine, pytest.raises(noulo.RequestError) as excinfo:
        engine.evaluate({"type": "noul", "input": "x"})
    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.message == "Noul requires a proposition."


def test_async_evaluate():
    with Noulo(engine=fake_engine()) as engine:
        assert asyncio.run(engine.aevaluate(CHOICE)) == {"type": "choice", "value": "A"}


def test_context_manager_releases_model():
    backend = FakeBackend()
    with Noulo(
        engine=DecisionEngine(
            load_backend=FakeLoader(**{"fake-model": backend}), model_id="fake-model"
        )
    ):
        pass
    assert backend.closed


def test_python_and_rest_apis_share_one_implementation(tmp_path):
    engine = fake_engine()
    app = create_app(
        Settings(_env_file=None, ui_enabled=False, learning_enabled=False, models_file=None),
        engine=engine,
        registry=ModelRegistry(tmp_path, None),
    )
    with TestClient(app) as client:
        embedded = Noulo(engine=engine)
        for request in (NOUL, CHOICE):
            assert client.post("/api/v1/evaluate", json=request).json() == embedded.evaluate(
                request
            )


def test_module_level_evaluate_uses_one_lazily_loaded_default(monkeypatch):
    created = []

    def factory():
        created.append(Noulo(engine=fake_engine()))
        return created[-1]

    monkeypatch.setattr("noulo.embedded._default", None)
    monkeypatch.setattr("noulo.embedded._make_default", factory)
    assert noulo.evaluate(NOUL)["value"] == pytest.approx(0.96)
    assert noulo.evaluate(CHOICE)["value"] == "A"
    assert asyncio.run(noulo.aevaluate(NOUL))["type"] == "noul"
    assert len(created) == 1
    noulo.close()
