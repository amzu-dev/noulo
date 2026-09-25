import json

import pytest

from noulo.teaching import TeachingError, batches, load_examples, to_request

NOUL = {
    "type": "noul",
    "input": "Unpaid for 120 days.",
    "proposition": "Payment is overdue.",
    "expected": True,
}
CHOICE = {
    "type": "choice",
    "input": "Charged twice.",
    "question": "Which team?",
    "choices": [{"id": "A", "text": "Billing"}, {"id": "B", "text": "Sales"}],
    "expected": "A",
}
SCORE = {
    "type": "score",
    "input": "All down.",
    "question": "How severe?",
    "rubric": ["low", "medium", "high"],
    "expected": "high",
}


# ---------------------------------------------------------------- reading files


def test_jsonl_file_skips_blank_lines_and_comments(tmp_path):
    path = tmp_path / "teach.jsonl"
    path.write_text("# my examples\n" + json.dumps(NOUL) + "\n\n" + json.dumps(CHOICE) + "\n")
    examples = load_examples(path)
    assert [(line, item["type"]) for line, item in examples] == [(2, "noul"), (4, "choice")]


def test_json_array_and_examples_object_are_accepted(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([NOUL, SCORE]))
    (tmp_path / "b.json").write_text(json.dumps({"examples": [CHOICE]}))
    assert len(load_examples(tmp_path / "a.json")) == 2
    assert load_examples(tmp_path / "b.json")[0][1]["type"] == "choice"


def test_malformed_line_reports_its_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(NOUL) + "\n{not json\n")
    with pytest.raises(TeachingError, match="line 2"):
        load_examples(path)


def test_missing_file_is_a_teaching_error(tmp_path):
    with pytest.raises(TeachingError, match="not found"):
        load_examples(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------- example -> (request, expected)


def test_expected_field_is_used_as_is():
    request, expected = to_request(CHOICE)
    assert request["type"] == "choice" and "expected" not in request and expected == "A"


@pytest.mark.parametrize("label,expected", [("yes", True), ("no", False), ("unknown", 0.5)])
def test_benchmark_dataset_noul_labels_are_understood(label, expected):
    item = {"input": "x", "proposition": "y", "label": label, "split": "test", "id": "n1"}
    request, value = to_request(item)
    assert request == {"type": "noul", "input": "x", "proposition": "y"} and value == expected


def test_benchmark_dataset_choice_answer_and_score_level_are_understood():
    choice = {k: v for k, v in CHOICE.items() if k not in ("type", "expected")} | {"answer": "B"}
    score = {k: v for k, v in SCORE.items() if k not in ("type", "expected")} | {
        "expected_level": 1
    }
    assert to_request(choice) == (
        {"type": "choice", **{k: choice[k] for k in ("input", "question", "choices")}},
        "B",
    )
    assert to_request(score)[1] == "medium"


def test_type_is_inferred_from_the_fields():
    assert to_request({"input": "x", "proposition": "y", "expected": 1})[0]["type"] == "noul"


def test_example_without_an_answer_is_rejected():
    with pytest.raises(TeachingError, match="expected"):
        to_request({"type": "noul", "input": "x", "proposition": "y"})


def test_example_of_unknown_shape_is_rejected():
    with pytest.raises(TeachingError, match="type"):
        to_request({"input": "x", "expected": 1})


# ---------------------------------------------------------------- batching for the API body limit


def test_batches_stay_under_the_byte_limit_and_keep_order():
    items = [{"i": n, "input": "x" * 100} for n in range(50)]
    chunks = list(batches(items, max_bytes=1000))
    assert [item["i"] for chunk in chunks for item in chunk] == list(range(50))
    assert all(len(json.dumps({"items": chunk})) <= 1000 for chunk in chunks)


def test_an_item_bigger_than_the_limit_goes_alone():
    chunks = list(batches([{"input": "x" * 5000}, {"input": "y"}], max_bytes=1000))
    assert [len(chunk) for chunk in chunks] == [1, 1]
