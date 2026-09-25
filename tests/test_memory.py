import json
import threading
from datetime import datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from noulo.inference.embedder import HashingEmbedder
from noulo.inference.memory import (
    DEFAULT_CONFIG,
    LearningMemory,
    MemoryConfig,
    task_key,
)
from noulo.inference.stores import open_store
from noulo.inference.types import ChoiceOption, Recollection

NOUL_TASK = task_key("noul", proposition="The invoice is overdue")
INVOICE = "Invoice #4411 was due on 1 March and has not been paid."
ROUTING_OPTIONS = {"billing": "Billing team", "it": "IT support", "sales": "Sales"}
CHOICE_TASK = task_key("choice", question="Which team should handle this?", options=ROUTING_OPTIONS)
PRINTER = "The office printer on floor 3 is jammed again and shows error E04."


def _parse_utc(stamp: str) -> datetime:
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert parsed.utcoffset() == timedelta(0)
    return parsed


def test_task_key_noul_normalises_proposition():
    assert task_key("noul", proposition="  The  Invoice\tis\nPAID ") == "noul|the invoice is paid"


def test_task_key_choice_sorts_options_by_id_and_normalises_text():
    options = [ChoiceOption("b", "Billing  Issue"), ChoiceOption("a", " Tech Support")]
    key = task_key("choice", question=" Which  TEAM? ", options=options)
    assert key == "choice|which team?|a=tech support;b=billing issue"
    assert task_key("choice", question="which team?", options=list(reversed(options))) == key
    assert (
        task_key(
            "choice", question="which team?", options={"b": "billing issue", "a": "tech support"}
        )
        == key
    )


def test_task_key_score_preserves_rubric_order():
    key = task_key("score", question="How  urgent?", rubric=["Low", " Medium ", "HIGH"])
    assert key == "score|how urgent?|low;medium;high"
    assert task_key("score", question="how urgent?", rubric=["high", "medium", "low"]) != key


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"primitive": "noul"}, id="noul-no-proposition"),
        pytest.param({"primitive": "noul", "proposition": "   "}, id="noul-blank"),
        pytest.param({"primitive": "choice", "options": {"a": "x"}}, id="choice-no-question"),
        pytest.param({"primitive": "choice", "question": "q"}, id="choice-no-options"),
        pytest.param({"primitive": "choice", "question": "q", "options": []}, id="choice-empty"),
        pytest.param({"primitive": "score", "rubric": ["lo", "hi"]}, id="score-no-question"),
        pytest.param({"primitive": "score", "question": "q"}, id="score-no-rubric"),
        pytest.param({"primitive": "score", "question": "q", "rubric": []}, id="score-empty"),
        pytest.param({"primitive": "vote", "question": "q"}, id="unknown-primitive"),
    ],
)
def test_task_key_rejects_missing_parts(kwargs):
    primitive = kwargs.pop("primitive")
    with pytest.raises(ValueError):
        task_key(primitive, **kwargs)


# --- LearningMemory --------------------------------------------------------


# In-memory and file-backed locations per store kind.
_EPHEMERAL = {"sqlite": ":memory:", "qdrant": ":memory:", "chroma": None}
_PERSISTENT = {"sqlite": "mem.sqlite3", "qdrant": "qdrant", "chroma": "chroma"}


@pytest.fixture(params=["sqlite", "qdrant", "chroma"])
def store_kind(request) -> str:
    return request.param


@pytest.fixture
def make_memory(store_kind, tmp_path):
    """Factory for memories over the store under test; all are closed afterwards."""
    made: list[LearningMemory] = []

    def _make(
        config: MemoryConfig = DEFAULT_CONFIG,
        *,
        embedder=None,
        persistent: bool = False,
    ) -> LearningMemory:
        location = str(tmp_path / _PERSISTENT[store_kind]) if persistent else _EPHEMERAL[store_kind]
        store = open_store(store_kind, location=location)
        memory = LearningMemory(store, embedder or HashingEmbedder(), config)
        made.append(memory)
        return memory

    yield _make
    for memory in made:
        memory.close()


@pytest.fixture
def mem(make_memory):
    return make_memory()


def test_record_then_recall_returns_observed_neighbour(mem):
    rid = mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    [hit] = mem.recall(primitive="noul", task=NOUL_TASK, input=INVOICE)
    assert hit.record_id == rid
    assert hit.similarity == pytest.approx(1.0, abs=1e-5)
    assert hit.verified is False
    assert hit.value == pytest.approx(0.9)
    assert hit.choice_id is None


def test_recall_ignores_inputs_below_min_similarity(mem):
    mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    unrelated = "The quarterly marketing offsite moves to the lake house in June."
    assert mem.recall(primitive="noul", task=NOUL_TASK, input=unrelated) == []


def test_recall_is_isolated_by_task_and_primitive(mem):
    mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    other_task = task_key("noul", proposition="The customer is angry")
    assert mem.recall(primitive="noul", task=other_task, input=INVOICE) == []
    assert mem.recall(primitive="score", task=NOUL_TASK, input=INVOICE) == []


def test_recall_ignores_rows_from_a_different_embedder(make_memory):
    first = make_memory(embedder=HashingEmbedder(dim=64), persistent=True)
    first.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    first.close()

    second = make_memory(embedder=HashingEmbedder(dim=128), persistent=True)
    assert second.recall(primitive="noul", task=NOUL_TASK, input=INVOICE) == []


def test_recall_returns_top_k_by_similarity_descending(make_memory):
    memory = make_memory(MemoryConfig(top_k=2, min_similarity=0.5))
    variants = [INVOICE, INVOICE + " Please advise.", INVOICE + " Please advise urgently today."]
    ids = [
        memory.record(primitive="noul", task=NOUL_TASK, input=text, model_id="m1", value=0.9)
        for text in reversed(variants)
    ]
    hits = memory.recall(primitive="noul", task=NOUL_TASK, input=INVOICE)
    assert [h.record_id for h in hits] == [ids[2], ids[1]]
    assert hits[0].similarity >= hits[1].similarity


@pytest.mark.parametrize(
    ("primitive", "value", "choice_id"),
    [
        ("noul", None, None),
        ("noul", 1.5, None),
        ("noul", -0.1, None),
        ("noul", float("nan"), None),
        ("noul", 0.5, "a"),
        ("score", None, "a"),
        ("choice", None, None),
        ("choice", None, ""),
        ("choice", None, "   "),
        ("choice", 0.5, "a"),
    ],
)
def test_record_validates_outcome_against_primitive(mem, primitive, value, choice_id):
    with pytest.raises(ValueError):
        mem.record(
            primitive=primitive,
            task="t",
            input=INVOICE,
            model_id="m1",
            value=value,
            choice_id=choice_id,
        )


def test_records_lists_json_friendly_rows_newest_first(mem):
    first = mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    second = mem.record(
        primitive="choice", task=CHOICE_TASK, input=PRINTER, model_id="m2", choice_id="it"
    )
    rows = mem.records()
    json.dumps(rows)
    assert [r["id"] for r in rows] == [second, first]
    newest, oldest = rows
    assert newest["observedChoice"] == "it"
    assert newest["observedValue"] is None
    assert oldest == {
        "id": first,
        "primitive": "noul",
        "task": NOUL_TASK,
        "input": INVOICE,
        "observedValue": 0.9,
        "observedChoice": None,
        "verifiedValue": None,
        "verifiedChoice": None,
        "hits": 1,
        "modelId": "m1",
        "createdAt": oldest["createdAt"],
        "updatedAt": oldest["updatedAt"],
    }
    assert _parse_utc(oldest["createdAt"]) <= _parse_utc(newest["createdAt"])
    _parse_utc(oldest["updatedAt"])


def test_records_supports_paging_and_primitive_filter(mem):
    ids = [
        mem.record(primitive="noul", task=NOUL_TASK, input=f"case {i}", model_id="m1", value=0.5)
        for i in range(5)
    ]
    mem.record(primitive="choice", task=CHOICE_TASK, input=PRINTER, model_id="m1", choice_id="it")
    page = mem.records(limit=2, offset=1, primitive="noul")
    assert [r["id"] for r in page] == [ids[3], ids[2]]
    assert {r["primitive"] for r in mem.records(primitive="choice")} == {"choice"}
    assert len(mem.records()) == 6


def test_record_deduplicates_same_case_and_model(mem):
    first = mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    before = mem.records()[0]
    again = mem.record(
        primitive="noul", task=NOUL_TASK, input="  " + INVOICE.upper(), model_id="m1", value=0.7
    )
    assert again == first
    [row] = mem.records()
    assert row["hits"] == 2
    assert row["observedValue"] == pytest.approx(0.7)
    assert row["input"] == before["input"]
    assert _parse_utc(row["updatedAt"]) >= _parse_utc(before["updatedAt"])
    assert row["createdAt"] == before["createdAt"]


def test_record_keeps_separate_rows_per_model(mem):
    a = mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    b = mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m2", value=0.8)
    assert a != b
    assert len(mem.records()) == 2


def test_memory_persists_through_the_given_store():
    store = open_store("sqlite", location=":memory:")
    memory = LearningMemory(store, HashingEmbedder())
    rid = memory.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    stored = store.get(rid)
    assert stored is not None
    assert stored.embedder_id == "hashing-256"
    assert stored.input_norm == " ".join(INVOICE.lower().split())
    memory.close()


def test_feedback_sets_verified_outcome_used_by_recall(mem):
    rid = mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    assert mem.feedback(rid, value=0.0) is True
    [hit] = mem.recall(primitive="noul", task=NOUL_TASK, input=INVOICE)
    assert hit.verified is True
    assert hit.value == pytest.approx(0.0)
    [row] = mem.records()
    assert row["observedValue"] == pytest.approx(0.9)
    assert row["verifiedValue"] == pytest.approx(0.0)


def test_feedback_on_choice_record(mem):
    rid = mem.record(
        primitive="choice", task=CHOICE_TASK, input=PRINTER, model_id="m1", choice_id="billing"
    )
    assert mem.feedback(rid, choice_id="it") is True
    [hit] = mem.recall(primitive="choice", task=CHOICE_TASK, input=PRINTER)
    assert (hit.verified, hit.choice_id, hit.value) == (True, "it", None)


def test_feedback_unknown_id_returns_false(mem):
    assert mem.feedback("does-not-exist", value=0.5) is False


def test_feedback_validates_outcome_against_record_primitive(mem):
    rid = mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    with pytest.raises(ValueError):
        mem.feedback(rid, choice_id="billing")
    with pytest.raises(ValueError):
        mem.feedback(rid, value=float("nan"))


def test_record_never_overwrites_verified_outcome(mem):
    rid = mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    mem.feedback(rid, value=0.0)
    assert (
        mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.95)
        == rid
    )
    [row] = mem.records()
    assert row["verifiedValue"] == pytest.approx(0.0)
    assert row["observedValue"] == pytest.approx(0.95)
    assert row["hits"] == 2


def test_teach_inserts_verified_record_without_model_output(mem):
    rid = mem.teach(primitive="noul", task=NOUL_TASK, input=INVOICE, value=0.0)
    [row] = mem.records()
    assert row["id"] == rid
    assert row["modelId"] == "feedback"
    assert row["verifiedValue"] == pytest.approx(0.0)
    assert row["observedValue"] is None
    [hit] = mem.recall(primitive="noul", task=NOUL_TASK, input=INVOICE)
    assert hit.verified is True


def test_teach_same_case_updates_the_taught_record(mem):
    first = mem.teach(primitive="choice", task=CHOICE_TASK, input=PRINTER, choice_id="billing")
    again = mem.teach(primitive="choice", task=CHOICE_TASK, input=PRINTER, choice_id="it")
    assert again == first
    [row] = mem.records()
    assert row["verifiedChoice"] == "it"


def test_teach_validates_outcome(mem):
    with pytest.raises(ValueError):
        mem.teach(primitive="score", task="t", input=INVOICE, value=2.0)


# --- blending ----------------------------------------------------------------


def _recollection(similarity=1.0, *, verified=True, value=None, choice_id=None, rid="r"):
    return Recollection(
        record_id=rid, similarity=similarity, verified=verified, value=value, choice_id=choice_id
    )


def test_feedback_on_duplicate_input_flips_a_wrong_noul(mem):
    rid = mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    mem.feedback(rid, value=0.0)
    neighbours = mem.recall(primitive="noul", task=NOUL_TASK, input=INVOICE)
    blended, adjustment = mem.blend_scalar(0.9, neighbours)
    assert blended < 0.5
    assert adjustment.applied is True
    assert adjustment.matches == 1
    assert adjustment.neighbours == neighbours


def test_blend_scalar_follows_the_design_formula(mem):
    # W = 1.0 * feedback_weight(1.0) = 1; alpha = 0.9 * 1 / (1 + 0.5) = 0.6
    blended, adjustment = mem.blend_scalar(0.9, [_recollection(1.0, value=0.0)])
    assert adjustment.influence == pytest.approx(0.6)
    assert blended == pytest.approx(0.4 * 0.9)


def test_blend_scalar_uses_similarity_weighted_mean(mem):
    neighbours = [
        _recollection(1.0, value=1.0, rid="a"),
        _recollection(0.8, verified=False, value=0.0, rid="b"),
    ]
    # w = [1.0, 0.8 * 0.25 = 0.2]; W = 1.2; mean = 1/1.2; alpha = 0.9 * 1.2 / 1.7
    alpha = 0.9 * 1.2 / 1.7
    blended, adjustment = mem.blend_scalar(0.5, neighbours)
    assert adjustment.influence == pytest.approx(alpha)
    assert adjustment.matches == 2
    assert blended == pytest.approx((1 - alpha) * 0.5 + alpha * (1.0 / 1.2))


def test_blend_scalar_without_neighbours_is_unchanged(mem):
    blended, adjustment = mem.blend_scalar(0.73, [])
    assert blended == pytest.approx(0.73)
    assert adjustment.applied is False
    assert adjustment.influence == 0.0
    assert adjustment.matches == 0


def test_unverified_matches_move_output_less_than_verified(mem):
    observed, _ = mem.blend_scalar(0.9, [_recollection(1.0, verified=False, value=0.0)])
    verified, _ = mem.blend_scalar(0.9, [_recollection(1.0, verified=True, value=0.0)])
    assert 0.9 > observed > verified


def test_dissimilar_inputs_do_not_affect_blend(mem):
    rid = mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    mem.feedback(rid, value=0.0)
    unrelated = "The quarterly marketing offsite moves to the lake house in June."
    neighbours = mem.recall(primitive="noul", task=NOUL_TASK, input=unrelated)
    blended, adjustment = mem.blend_scalar(0.9, neighbours)
    assert blended == pytest.approx(0.9)
    assert adjustment.applied is False


def test_different_task_keys_do_not_affect_blend(mem):
    mem.teach(primitive="noul", task=NOUL_TASK, input=INVOICE, value=0.0)
    other = task_key("noul", proposition="The customer is a VIP")
    neighbours = mem.recall(primitive="noul", task=other, input=INVOICE)
    blended, adjustment = mem.blend_scalar(0.9, neighbours)
    assert blended == pytest.approx(0.9)
    assert adjustment.applied is False


OPTION_IDS = ["billing", "it", "sales"]


def test_blend_distribution_follows_the_design_formula(mem):
    probs, adjustment = mem.blend_distribution(
        [0.7, 0.2, 0.1], OPTION_IDS, [_recollection(1.0, choice_id="it")]
    )
    # alpha = 0.6, q = [0, 1, 0]
    assert probs == pytest.approx([0.4 * 0.7, 0.4 * 0.2 + 0.6, 0.4 * 0.1])
    assert adjustment.applied is True
    assert adjustment.influence == pytest.approx(0.6)
    assert adjustment.matches == 1


def test_blend_distribution_ignores_unknown_option_ids(mem):
    neighbours = [
        _recollection(1.0, choice_id="legal", rid="a"),
        _recollection(1.0, choice_id="sales", rid="b"),
    ]
    probs, adjustment = mem.blend_distribution([0.7, 0.2, 0.1], OPTION_IDS, neighbours)
    assert len(probs) == 3
    assert adjustment.matches == 1
    assert [r.record_id for r in adjustment.neighbours] == ["b"]
    assert probs[2] > 0.1

    only_unknown, adjustment = mem.blend_distribution(
        [0.7, 0.2, 0.1], OPTION_IDS, [_recollection(1.0, choice_id="legal")]
    )
    assert only_unknown == pytest.approx([0.7, 0.2, 0.1])
    assert adjustment.applied is False


def test_blend_distribution_without_neighbours_is_unchanged(mem):
    probs, adjustment = mem.blend_distribution([0.5, 0.3, 0.2], OPTION_IDS, [])
    assert probs == pytest.approx([0.5, 0.3, 0.2])
    assert adjustment.applied is False


def test_blend_distribution_rejects_length_mismatch(mem):
    with pytest.raises(ValueError):
        mem.blend_distribution([0.5, 0.5], OPTION_IDS, [])


def test_taught_choice_changes_the_winner(mem):
    mem.teach(primitive="choice", task=CHOICE_TASK, input=PRINTER, choice_id="it")
    neighbours = mem.recall(primitive="choice", task=CHOICE_TASK, input=PRINTER)
    probs, _ = mem.blend_distribution([0.6, 0.3, 0.1], OPTION_IDS, neighbours)
    assert OPTION_IDS[max(range(3), key=probs.__getitem__)] == "it"
    assert sum(probs) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def blender():
    """Store-independent blending (hypothesis forbids function-scoped fixtures)."""
    memory = LearningMemory(open_store("sqlite", location=":memory:"), HashingEmbedder())
    yield memory
    memory.close()


unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
scalar_neighbours = st.lists(
    st.builds(
        Recollection,
        record_id=st.just("r"),
        similarity=unit,
        verified=st.booleans(),
        value=unit,
        choice_id=st.none(),
    ),
    max_size=8,
)


@given(model=unit, neighbours=scalar_neighbours)
def test_blended_scalar_stays_in_unit_interval(blender, model, neighbours):
    blended, adjustment = blender.blend_scalar(model, neighbours)
    assert 0.0 <= blended <= 1.0
    assert 0.0 <= adjustment.influence <= blender.config.max_influence


@st.composite
def distributions(draw):
    size = draw(st.integers(min_value=1, max_value=6))
    weights = draw(st.lists(unit, min_size=size, max_size=size))
    total = sum(weights)
    probs = [w / total for w in weights] if total > 0 else [1.0 / size] * size
    option_ids = [f"opt{i}" for i in range(size)]
    neighbours = draw(
        st.lists(
            st.builds(
                Recollection,
                record_id=st.just("r"),
                similarity=unit,
                verified=st.booleans(),
                value=st.none(),
                choice_id=st.sampled_from([*option_ids, "unknown"]),
            ),
            max_size=8,
        )
    )
    return probs, option_ids, neighbours


@given(case=distributions())
def test_blended_distribution_is_a_valid_distribution(blender, case):
    probs, option_ids, neighbours = case
    blended, adjustment = blender.blend_distribution(probs, option_ids, neighbours)
    assert len(blended) == len(option_ids)
    assert all(p >= 0.0 for p in blended)
    assert sum(blended) == pytest.approx(1.0)
    assert all(r.choice_id in option_ids for r in adjustment.neighbours)


# --- maintenance, persistence, concurrency -------------------------------------


def test_stats_counts_records_verified_and_names_embedder(mem):
    rid = mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    mem.record(primitive="choice", task=CHOICE_TASK, input=PRINTER, model_id="m1", choice_id="it")
    mem.teach(primitive="noul", task=NOUL_TASK, input="Paid in full.", value=0.0)
    mem.feedback(rid, value=0.0)
    assert mem.stats() == {"records": 3, "verified": 2, "embedder": "hashing-256"}


def test_clear_deletes_everything_and_returns_count(mem):
    mem.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    mem.teach(primitive="noul", task=NOUL_TASK, input="Paid in full.", value=0.0)
    assert mem.clear() == 2
    assert mem.records() == []
    assert mem.stats()["records"] == 0
    assert mem.recall(primitive="noul", task=NOUL_TASK, input=INVOICE) == []


class _ClosingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_close_closes_store_and_embedder_once(store_kind):
    embedder = _ClosingEmbedder()
    store = open_store(store_kind, location=_EPHEMERAL[store_kind])
    memory = LearningMemory(store, embedder)
    memory.close()
    memory.close()
    assert embedder.closed == 1


def test_learning_persists_across_reopening(make_memory):
    first = make_memory(persistent=True)
    rid = first.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
    first.feedback(rid, value=0.0)
    first.close()

    reopened = make_memory(persistent=True)
    neighbours = reopened.recall(primitive="noul", task=NOUL_TASK, input=INVOICE)
    assert [(n.record_id, n.verified) for n in neighbours] == [(rid, True)]
    assert reopened.blend_scalar(0.9, neighbours)[0] < 0.5
    assert reopened.stats()["verified"] == 1


def test_concurrent_records_from_eight_threads(make_memory):
    memory = make_memory(persistent=True)
    threads_count, per_thread = 8, 10
    errors: list[BaseException] = []
    start = threading.Barrier(threads_count)

    def worker(n: int) -> None:
        try:
            start.wait()
            for i in range(per_thread):
                memory.record(
                    primitive="noul",
                    task=NOUL_TASK,
                    input=f"thread {n} case {i}",
                    model_id="m1",
                    value=0.5,
                )
            memory.record(primitive="noul", task=NOUL_TASK, input=INVOICE, model_id="m1", value=0.9)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert memory.stats()["records"] == threads_count * per_thread + 1
    shared = [r for r in memory.records(limit=1000) if r["input"] == INVOICE]
    assert len(shared) == 1
    assert shared[0]["hits"] == threads_count
