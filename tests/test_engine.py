import math
import threading
import time

import pytest

from noulo.api.validation import ChoiceOption, ChoiceRequest, NoulRequest, ScoreRequest
from noulo.inference.calibration import PlattCalibrator
from noulo.inference.engine import (
    DecisionEngine,
    EngineBusy,
    EngineUnavailable,
    InferenceError,
)
from noulo.inference.types import BackendError, ModelLoadError
from tests.fakes import FakeBackend, FakeLoader

OPTIONS = [ChoiceOption(id="A", text="Billing"), ChoiceOption(id="B", text="Sales")]


def started(backend=None, **kwargs) -> DecisionEngine:
    loader = FakeLoader(**{"fake-model": backend or FakeBackend()})
    engine = DecisionEngine(load_backend=loader, model_id="fake-model", **kwargs)
    engine.start()
    return engine


# ------------------------------------------------------------------ lifecycle


def test_start_loads_model_once_and_runs_readiness_check():
    backend = FakeBackend()
    loader = FakeLoader(**{"fake-model": backend})
    engine = DecisionEngine(load_backend=loader, model_id="fake-model")
    engine.start()
    for _ in range(5):
        engine.noul("x", "y")
    assert loader.loads == ["fake-model"]
    assert backend.warmed
    assert engine.state == "ready" and engine.ready


def test_evaluating_before_start_is_rejected_as_not_ready():
    engine = DecisionEngine(load_backend=FakeLoader(), model_id="fake-model")
    with pytest.raises(EngineUnavailable) as excinfo:
        engine.noul("x", "y")
    assert excinfo.value.code == "MODEL_NOT_READY"


def test_model_load_failure_marks_engine_failed():
    engine = DecisionEngine(load_backend=FakeLoader(), model_id="missing")
    with pytest.raises(ModelLoadError):
        engine.start()
    assert engine.state == "failed" and not engine.ready


def test_readiness_check_failure_closes_backend_and_fails():
    backend = FakeBackend(warmup_error=ModelLoadError("bad output"))
    engine = DecisionEngine(
        load_backend=FakeLoader(**{"fake-model": backend}), model_id="fake-model"
    )
    with pytest.raises(ModelLoadError):
        engine.start()
    assert backend.closed
    assert engine.state == "failed"


def test_model_info_is_exposed():
    assert started().model_info.id == "fake-model"


# ------------------------------------------------------------------ primitives


def test_noul_applies_calibration_layer():
    engine = started(
        FakeBackend(noul_value=0.8), load_calibrator=lambda _: PlattCalibrator(1.0, -1.0)
    )
    assert engine.noul("x", "y").value == pytest.approx(PlattCalibrator(1.0, -1.0)(0.8))


def test_choice_returns_supplied_id_with_highest_probability():
    engine = started(FakeBackend(choice_probs=[0.3, 0.7]))
    result = engine.choice("x", "q?", OPTIONS)
    assert (result.type, result.value) == ("choice", "B")


def test_score_returns_expected_level_normalised():
    engine = started(FakeBackend(score_probs=[0.0, 0.5, 0.5]))
    assert engine.score("x", "q?", ["low", "mid", "high"]).value == pytest.approx(0.75)


def test_evaluate_dispatches_parsed_requests():
    engine = started(FakeBackend(noul_value=0.9, choice_probs=[0.9, 0.1], score_probs=[0, 1]))
    assert engine.evaluate(NoulRequest(input="x", proposition="y")).type == "noul"
    assert engine.evaluate(ChoiceRequest(input="x", question="q", choices=OPTIONS)).value == "A"
    assert engine.evaluate(ScoreRequest(input="x", question="q", rubric=["a", "b"])).value == 1.0


def test_diagnostics_include_probabilities_and_model():
    engine = started(FakeBackend(choice_probs=[0.25, 0.75]))
    result = engine.choice("x", "q?", OPTIONS, diagnostics=True)
    assert result.diagnostics["model"] == "fake-model"
    assert result.diagnostics["probabilities"] == {"A": 0.25, "B": 0.75}


def test_results_without_diagnostics_carry_none():
    assert started().noul("x", "y").diagnostics is None


# ------------------------------------------------------------------ invariant guards


@pytest.mark.parametrize("bad", [math.nan, 1.5, -0.1, math.inf])
def test_out_of_range_noul_from_backend_is_an_inference_error(bad):
    engine = started(FakeBackend(noul_value=bad))
    with pytest.raises(InferenceError):
        engine.noul("x", "y")


def test_wrong_length_choice_distribution_is_an_inference_error():
    engine = started(FakeBackend(choice_probs=[1.0]))
    with pytest.raises(InferenceError):
        engine.choice("x", "q?", OPTIONS)


def test_backend_failure_is_wrapped_and_flags_remote_backends():
    local = started(FakeBackend(error=BackendError("boom")))
    with pytest.raises(InferenceError) as excinfo:
        local.noul("x", "y")
    assert excinfo.value.remote is False

    remote = started(FakeBackend(local=False, error=BackendError("upstream 500")))
    with pytest.raises(InferenceError) as excinfo:
        remote.noul("x", "y")
    assert excinfo.value.remote is True


def test_unexpected_backend_exception_is_wrapped():
    engine = started(FakeBackend(error=RuntimeError("segfault-ish")))
    with pytest.raises(InferenceError):
        engine.noul("x", "y")


# ------------------------------------------------------------------ concurrency


def test_concurrency_limit_serialises_backend_calls():
    backend = FakeBackend(delay=0.02)
    engine = started(backend, max_concurrency=1, max_queue=16)
    threads = [threading.Thread(target=engine.noul, args=("x", "y")) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert backend.calls == 8
    assert backend.max_active == 1


def test_queue_limit_rejects_excess_requests_as_busy():
    gate = threading.Event()
    backend = FakeBackend(gate=gate)
    engine = started(backend, max_concurrency=1, max_queue=0)
    worker = threading.Thread(target=engine.noul, args=("x", "y"))
    worker.start()
    assert backend.started.wait(2)
    with pytest.raises(EngineBusy):
        engine.noul("x", "y")
    gate.set()
    worker.join()


# ------------------------------------------------------------------ model switching


def test_switch_model_activates_new_backend_and_closes_old():
    old, new = FakeBackend("m1"), FakeBackend("m2", noul_value=0.1)
    engine = DecisionEngine(load_backend=FakeLoader(m1=old, m2=new), model_id="m1")
    engine.start()
    info = engine.switch_model("m2")
    assert info.id == "m2" and engine.model_info.id == "m2"
    assert new.warmed and old.closed
    assert engine.noul("x", "y").value == pytest.approx(0.1)


def test_switch_model_failure_keeps_current_model():
    old = FakeBackend("m1")
    engine = DecisionEngine(load_backend=FakeLoader(m1=old), model_id="m1")
    engine.start()
    with pytest.raises(ModelLoadError):
        engine.switch_model("nope")
    assert engine.model_info.id == "m1" and not old.closed
    assert engine.noul("x", "y").value == pytest.approx(0.8)


def test_in_flight_request_finishes_on_old_model_during_switch():
    gate = threading.Event()
    old, new = FakeBackend("m1", noul_value=0.9, gate=gate), FakeBackend("m2", noul_value=0.2)
    engine = DecisionEngine(
        load_backend=FakeLoader(m1=old, m2=new), model_id="m1", max_concurrency=2
    )
    engine.start()
    results = []
    worker = threading.Thread(target=lambda: results.append(engine.noul("x", "y").value))
    worker.start()
    assert old.started.wait(2)
    engine.switch_model("m2")
    assert not old.closed  # still serving the in-flight request
    gate.set()
    worker.join()
    assert results == [pytest.approx(0.9)]
    assert old.closed


def test_switch_model_loads_calibration_for_new_model():
    calibrators = {"m1": PlattCalibrator(1.0, 0.0), "m2": PlattCalibrator(1.0, 2.0)}
    engine = DecisionEngine(
        load_backend=FakeLoader(m1=FakeBackend("m1"), m2=FakeBackend("m2")),
        model_id="m1",
        load_calibrator=calibrators.__getitem__,
    )
    engine.start()
    engine.switch_model("m2")
    assert engine.noul("x", "y").value == pytest.approx(PlattCalibrator(1.0, 2.0)(0.8))


# ------------------------------------------------------------------ shutdown


def test_shutdown_waits_for_in_flight_then_releases_model():
    gate = threading.Event()
    backend = FakeBackend(gate=gate)
    engine = started(backend, max_concurrency=2)
    worker = threading.Thread(target=engine.noul, args=("x", "y"))
    worker.start()
    assert backend.started.wait(2)

    done = []
    stopper = threading.Thread(target=lambda: done.append(engine.shutdown(timeout=5)))
    stopper.start()
    time.sleep(0.05)
    assert not backend.closed  # still waiting for the in-flight request
    with pytest.raises(EngineUnavailable) as excinfo:
        engine.noul("x", "y")  # no new work while stopping
    assert excinfo.value.code == "SHUTTING_DOWN"

    gate.set()
    worker.join()
    stopper.join()
    assert done == [True]
    assert backend.closed and engine.state == "stopped"


def test_shutdown_timeout_reports_unclean_drain_but_still_stops():
    gate = threading.Event()
    backend = FakeBackend(gate=gate)
    engine = started(backend)
    worker = threading.Thread(target=engine.noul, args=("x", "y"))
    worker.start()
    assert backend.started.wait(2)
    assert engine.shutdown(timeout=0.05) is False
    assert engine.state == "stopped"
    gate.set()
    worker.join()


def test_shutdown_is_idempotent():
    engine = started()
    assert engine.shutdown() is True
    assert engine.shutdown() is True
