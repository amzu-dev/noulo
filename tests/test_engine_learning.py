import pytest

from noulo.api.validation import ChoiceOption, ChoiceRequest, NoulRequest, ScoreRequest
from noulo.inference.embedder import HashingEmbedder
from noulo.inference.engine import DecisionEngine, EngineUnavailable, RecordNotFound
from noulo.inference.memory import LearningMemory
from noulo.inference.stores import open_store
from tests.fakes import FakeBackend, FakeLoader

OPTIONS = [ChoiceOption(id="A", text="Billing"), ChoiceOption(id="B", text="Sales")]
INPUT = "I was charged twice for my subscription this month."


def open_memory():
    return LearningMemory(open_store("sqlite", location=":memory:"), HashingEmbedder())


def engine_with(backend=None, *, learning=True, memory_factory=open_memory):
    engine = DecisionEngine(
        load_backend=FakeLoader(**{"fake-model": backend or FakeBackend()}),
        model_id="fake-model",
        open_memory=memory_factory,
        learning=learning,
    )
    engine.start()
    return engine


def test_evaluations_are_recorded_with_a_record_id():
    engine = engine_with()
    result = engine.noul(INPUT, "The customer was double charged.")
    assert result.record_id
    assert engine.memory.stats()["records"] == 1


def test_repeated_identical_evaluations_are_deduplicated():
    engine = engine_with()
    first = engine.noul(INPUT, "p").record_id
    second = engine.noul(INPUT, "p").record_id
    assert first == second and engine.memory.stats()["records"] == 1


def test_noul_feedback_corrects_future_answers_for_the_same_case():
    engine = engine_with(FakeBackend(noul_value=0.9))
    record_id = engine.noul(INPUT, "The customer is happy.").record_id
    engine.feedback(record_id, False)
    assert engine.noul(INPUT, "The customer is happy.").value < 0.5


def test_choice_feedback_changes_the_selected_supplied_id():
    engine = engine_with(FakeBackend(choice_probs=[0.8, 0.2]))
    record_id = engine.choice(INPUT, "Which team?", OPTIONS).record_id
    engine.feedback(record_id, "B")
    assert engine.choice(INPUT, "Which team?", OPTIONS).value == "B"


def test_score_feedback_moves_the_score():
    engine = engine_with(FakeBackend(score_probs=[0.0, 0.0, 1.0]))
    record_id = engine.score(INPUT, "How bad?", ["low", "mid", "high"]).record_id
    engine.feedback(record_id, 0.0)
    assert engine.score(INPUT, "How bad?", ["low", "mid", "high"]).value < 0.5


def test_diagnostics_report_learning_influence():
    engine = engine_with(FakeBackend(noul_value=0.9))
    record_id = engine.noul(INPUT, "p").record_id
    engine.feedback(record_id, 0.0)
    learning = engine.noul(INPUT, "p", diagnostics=True).diagnostics["learning"]
    assert learning["applied"] is True and learning["influence"] > 0 and learning["matches"] >= 1


def test_teach_with_a_full_request_influences_similar_inputs():
    engine = engine_with(FakeBackend(noul_value=0.9))
    engine.teach(NoulRequest(input=INPUT, proposition="The customer is happy."), 0.0)
    assert engine.noul(INPUT, "The customer is happy.").value < 0.5


def test_teach_choice_rejects_ids_that_were_not_supplied():
    engine = engine_with()
    with pytest.raises(ValueError, match="supplied"):
        engine.teach(ChoiceRequest(input=INPUT, question="q", choices=OPTIONS), "Z")


def test_teach_score_accepts_rubric_level_names():
    engine = engine_with(FakeBackend(score_probs=[0.0, 0.0, 1.0]))
    request = ScoreRequest(input=INPUT, question="How bad?", rubric=["low", "mid", "high"])
    engine.teach(request, "low")
    assert engine.score(INPUT, "How bad?", ["low", "mid", "high"]).value < 0.5


def test_feedback_with_wrong_outcome_kind_is_rejected():
    engine = engine_with()
    record_id = engine.noul(INPUT, "p").record_id
    with pytest.raises(ValueError):
        engine.feedback(record_id, "A")


def test_feedback_for_unknown_record_raises():
    engine = engine_with()
    with pytest.raises(RecordNotFound):
        engine.feedback("nope", 1.0)


def test_learning_disabled_records_nothing_and_rejects_feedback():
    engine = engine_with(learning=False)
    assert engine.noul(INPUT, "p").record_id is None
    assert engine.memory is None  # not even loaded: saves RAM
    with pytest.raises(EngineUnavailable) as excinfo:
        engine.feedback("any", 1.0)
    assert excinfo.value.code == "LEARNING_DISABLED"


def test_learning_can_be_switched_on_and_off_at_runtime():
    engine = engine_with(learning=False)
    engine.set_learning(True)
    assert engine.noul(INPUT, "p").record_id is not None
    engine.set_learning(False)
    assert engine.noul(INPUT, "p").record_id is None


def test_learning_failure_never_breaks_inference():
    class BrokenMemory:
        def recall(self, **_):
            raise RuntimeError("store offline")

        def record(self, **_):
            raise RuntimeError("store offline")

        def close(self):
            pass

    engine = engine_with(FakeBackend(noul_value=0.7), memory_factory=BrokenMemory)
    result = engine.noul(INPUT, "p")
    assert result.value == pytest.approx(0.7) and result.record_id is None


def test_shutdown_closes_memory():
    closed = []
    memory = open_memory()
    original = memory.close
    memory.close = lambda: (closed.append(True), original())
    engine = engine_with(memory_factory=lambda: memory)
    engine.shutdown()
    assert closed == [True]
