import json

import pytest

from noulo.benchmark.data import load_dataset


def test_load_dataset_rejects_paraphrase_group_leakage_before_filtering(tmp_path):
    path = tmp_path / "choice.jsonl"
    rows = [
        {
            "id": "one",
            "group": "duplicate-payment",
            "split": "calibration",
            "input": "Charged twice",
        },
        {"id": "two", "group": "duplicate-payment", "split": "test", "input": "A duplicate charge"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="group.*duplicate-payment.*splits"):
        load_dataset(path, split="test")


def test_same_group_can_have_multiple_examples_in_one_split(tmp_path):
    path = tmp_path / "choice.jsonl"
    rows = [
        {"id": "one", "group": "billing", "split": "test"},
        {"id": "two", "group": "billing", "split": "test"},
        {"id": "three", "split": "calibration"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert load_dataset(path, split="test") == rows[:2]
    assert load_dataset(path) == rows


@pytest.mark.parametrize("group", ["", "   ", [], {}, 42])
def test_invalid_group_is_rejected_clearly(tmp_path, group):
    path = tmp_path / "choice.jsonl"
    path.write_text(json.dumps({"id": "one", "group": group, "split": "test"}))
    with pytest.raises(ValueError, match="group must be a non-empty string"):
        load_dataset(path)
