import json

import numpy as np
import pytest

from noulo.benchmark.data import load_dataset
from noulo.benchmark.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    mean_absolute_error,
    percentile_ms,
)
from noulo.benchmark.report import format_report
from noulo.benchmark.tune import best_strategy, choose_calibration_method


def test_accuracy():
    assert accuracy(["A", "B", "C"], ["A", "B", "D"]) == pytest.approx(2 / 3)


def test_accuracy_of_nothing_is_nan():
    assert np.isnan(accuracy([], []))


def test_brier_score_perfect_and_worst():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_ece_is_zero_for_perfectly_calibrated_bins():
    # 10 predictions of 0.7 with 7 positives -> perfectly calibrated
    assert expected_calibration_error([0.7] * 10, [1] * 7 + [0] * 3) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    assert expected_calibration_error([0.99] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.49)


def test_mae():
    assert mean_absolute_error([0.0, 1.0], [0.5, 0.5]) == pytest.approx(0.5)


def test_percentiles_in_milliseconds():
    seconds = [0.001 * i for i in range(1, 101)]
    assert percentile_ms(seconds, 50) == pytest.approx(50.5, abs=0.5)
    assert percentile_ms(seconds, 95) == pytest.approx(95.05, abs=0.5)


def test_load_dataset_filters_split(tmp_path):
    path = tmp_path / "noul.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"id": "1", "split": "test"},
                {"id": "2", "split": "calibration"},
                {"id": "3", "split": "test"},
            ]
        )
        + "\n"
    )
    assert [r["id"] for r in load_dataset(path, split="test")] == ["1", "3"]
    assert len(load_dataset(path)) == 3


def test_real_dataset_files_are_present_and_split():
    for name in ("noul", "choice", "score"):
        rows = load_dataset(f"benchmark/data/{name}.jsonl")
        assert {r["split"] for r in rows} == {"calibration", "test"}


def test_best_strategy_minimises_loss_and_prefers_first_on_ties():
    losses = {"a": 0.3, "b": 0.1, "c": 0.1}
    assert best_strategy(losses) == "b"


def test_choose_calibration_method_picks_lowest_cross_validated_log_loss():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.2, 0.8, 300)
    y = (rng.uniform(0, 1, 300) < true_p).astype(float)
    raw = 1 / (1 + np.exp(-np.log(true_p / (1 - true_p)) * 4))  # overconfident
    method, scores = choose_calibration_method(raw, y)
    assert method != "identity"
    assert scores[method] <= scores["identity"]


def test_report_has_spec_fields_and_measured_values():
    text = format_report(
        {
            "model": "m",
            "quantization": "INT8",
            "modelSizeBytes": 91_000_000,
            "peakRssBytes": 300_000_000,
            "coldStartMs": 812.3,
            "choiceAccuracy": 0.9,
            "noulAccuracy": 0.85,
            "noulEce": 0.05,
            "noulBrier": 0.1,
            "scoreMae": 0.12,
            "p50Ms": 20.0,
            "p95Ms": 45.0,
            "throughputRps": 30.0,
        }
    )
    for label in (
        "Model:",
        "Quantisation:",
        "Model size:",
        "Peak RAM:",
        "Cold startup:",
        "Choice accuracy:",
        "Noul accuracy:",
        "Noul calibration:",
        "Score MAE:",
        "P50 inference:",
        "P95 inference:",
    ):
        assert label in text
    assert "86.8 MiB" in text and "90.0%" in text


def test_write_measured_keeps_the_numbers_pickers_show(tmp_path):
    from noulo.benchmark.run import write_measured

    result = {
        "model": "m",
        "peakRssBytes": 1,
        "modelSizeBytes": 2,
        "noulAccuracy": 0.9,
        "choiceAccuracy": 0.7,
        "scoreMae": 0.2,
        "p50Ms": 8.0,
        "p95Ms": 30.0,
        "coldStartMs": 500.0,
        "noulEce": 0.1,
        "items": {"noul": 1},
    }
    write_measured(tmp_path / "m", result, machine="Apple M1 Pro")
    saved = json.loads((tmp_path / "m" / "measured.json").read_text())
    assert saved["peakRssBytes"] == 1 and saved["noulAccuracy"] == 0.9
    assert saved["machine"] == "Apple M1 Pro" and "items" not in saved


def test_low_memory_runs_do_not_overwrite_default_measurements(tmp_path, monkeypatch):
    from noulo.benchmark import run

    written = []
    monkeypatch.setattr(
        run, "_run_worker", lambda model_id, args: {"model": model_id, "peakRssBytes": 1}
    )
    monkeypatch.setattr(run, "write_measured", lambda *a, **k: written.append(a))
    monkeypatch.setattr(run, "format_report", lambda r: "")
    models = tmp_path / "models"
    args = ["--model", "nli-mobilebert-int8", "--models-dir", str(models), "--out", str(tmp_path)]
    (models / "nli-mobilebert-int8").mkdir(parents=True)
    for name in ("model.onnx", "tokenizer.json", "config.json"):
        (models / "nli-mobilebert-int8" / name).write_text("{}")
    run.main([*args, "--low-memory"])
    assert written == []
    run.main(args)
    assert len(written) == 1
