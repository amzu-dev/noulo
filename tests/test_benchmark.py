import json
from types import SimpleNamespace

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


@pytest.mark.parametrize(
    ("predicted", "expected", "summary"),
    [
        (
            ["billing", "sales", "sales", "sales"],
            ["billing", "billing", "sales", "billing"],
            {
                "correct": 2,
                "total": 4,
                "confusion": {"billing": {"billing": 1, "sales": 2}, "sales": {"sales": 1}},
            },
        ),
        ([], [], {"correct": 0, "total": 0, "confusion": {}}),
    ],
)
def test_choice_summary_counts_correct_total_and_expected_to_predicted_ids(
    predicted, expected, summary
):
    from noulo.benchmark import metrics

    assert metrics.choice_summary(predicted, expected) == summary


@pytest.mark.parametrize("predicted, expected", [(["A"], []), ([], ["A"])])
def test_choice_summary_rejects_mismatched_prediction_counts(predicted, expected):
    from noulo.benchmark.metrics import choice_summary

    with pytest.raises(ValueError, match="same length"):
        choice_summary(predicted, expected)


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


@pytest.fixture
def choice_benchmark(tmp_path, monkeypatch):
    from noulo import runtime
    from noulo.inference.calibration import IdentityCalibrator
    from tests.fakes import FakeBackend, FakeLoader

    backend = FakeBackend(choice_probs=[0.25, 0.75])
    registry = SimpleNamespace(
        load_backend=FakeLoader(**{"fake-model": backend}),
        load_calibrator=lambda _: IdentityCalibrator(),
        list=lambda: [{"id": "fake-model", "installed": True}],
    )
    monkeypatch.setattr(runtime, "build_registry", lambda _: registry)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    choices = [
        {"id": "route/billing", "text": "private billing option"},
        {"id": "route/sales", "text": "private sales option"},
    ]
    rows = [
        {
            "id": id,
            "split": split,
            "input": "private customer message",
            "question": "private routing question",
            "choices": choices,
            "answer": answer,
        }
        for id, split, answer in [
            ("right", "test", "route/sales"),
            ("wrong", "test", "route/billing"),
            ("calibration-only", "calibration", "route/billing"),
        ]
    ]
    (data_dir / "choice.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for name in ("noul", "score"):
        (data_dir / f"{name}.jsonl").write_text("")
    return data_dir, backend


def test_choice_benchmark_accepts_legacy_examples_without_ids(choice_benchmark, tmp_path):
    from noulo.benchmark.run import measure

    data_dir, _ = choice_benchmark
    path = data_dir / "choice.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row.pop("id")
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = measure("fake-model", tmp_path / "models", None, data_dir)
    assert [row["id"] for row in result["choicePredictions"]] == [None, None]
    assert result["choiceSummary"]["total"] == 2


@pytest.mark.parametrize("entry", ["measure", "worker", "main"])
@pytest.mark.parametrize("split", ["test", "calibration"])
def test_choice_benchmark_json_has_text_free_predictions_and_counts(
    choice_benchmark, tmp_path, monkeypatch, capsys, entry, split
):
    from noulo.benchmark import run

    data_dir, backend = choice_benchmark
    models_dir = tmp_path / "models"
    out = tmp_path / "results"
    args = [
        "--models-dir",
        str(models_dir),
        "--data-dir",
        str(data_dir),
        "--split",
        split,
        "--out",
        str(out),
    ]
    if entry == "measure":
        result = run.measure("fake-model", models_dir, None, data_dir, split=split)
    elif entry == "worker":
        assert run.main([*args, "--worker", "fake-model"]) == 0
        result = run.parse_worker_output(capsys.readouterr().out)
    else:
        monkeypatch.setattr(
            run,
            "_run_worker",
            lambda model_id, args: run.measure(
                model_id, args.models_dir, args.models_file, args.data_dir, args.split
            ),
        )
        assert run.main([*args, "--model", "fake-model"]) == 0
        result = json.loads((out / "fake-model.json").read_text())

    expected_items = (
        [("right", "route/sales", True), ("wrong", "route/billing", False)]
        if split == "test"
        else [("calibration-only", "route/billing", False)]
    )
    assert result["choicePredictions"] == [
        {
            "id": id,
            "expectedId": expected,
            "predictedId": "route/sales",
            "correct": correct,
            "probabilities": {"route/billing": 0.25, "route/sales": 0.75},
        }
        for id, expected, correct in expected_items
    ]
    confusion = {"route/billing": {"route/sales": 1}}
    if split == "test":
        confusion["route/sales"] = {"route/sales": 1}
    assert result["choiceSummary"] == {
        "correct": int(split == "test"),
        "total": len(expected_items),
        "confusion": confusion,
    }
    assert result["choiceAccuracy"] == (0.5 if split == "test" else 0.0)
    assert result["items"]["choice"] == len(expected_items)
    assert result["split"] == split
    assert "private" not in json.dumps(result)
    assert backend.calls == len(expected_items)
    assert backend.warmed and backend.closed


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    ("dataset", "split", "mode", "should_write"),
    [
        ("default", "test", [], True),
        ("absolute-default", "test", [], True),
        ("default", "calibration", [], False),
        ("custom", "test", [], False),
        ("custom", "calibration", [], False),
        ("default", "test", ["--device", "coreml"], False),
        ("default", "test", ["--low-memory"], False),
    ],
)
def test_generic_measurements_only_written_for_bundled_test_cpu_baseline(
    choice_benchmark, tmp_path, monkeypatch, existing, dataset, split, mode, should_write
):
    from noulo.benchmark import run

    data_dir, _ = choice_benchmark
    result = {"model": "fake-model", "choiceAccuracy": 0.5, "split": split}
    monkeypatch.setattr(run, "_run_worker", lambda model_id, args: result)
    monkeypatch.setattr(run, "machine_name", lambda: "test machine")
    models_dir = tmp_path / "models"
    measured = models_dir / "fake-model" / "measured.json"
    original = '{"choiceAccuracy": 0.9, "machine": "existing baseline"}\n'
    if existing:
        measured.parent.mkdir(parents=True)
        measured.write_text(original)
    out = tmp_path / "results"
    args = [
        "--model",
        "fake-model",
        "--models-dir",
        str(models_dir),
        "--out",
        str(out),
        "--split",
        split,
        *mode,
    ]
    if dataset == "absolute-default":
        args += ["--data-dir", str(run.DEFAULT_DATA_DIR.resolve())]
    elif dataset == "custom":
        args += ["--data-dir", str(data_dir)]

    assert run.main(args) == 0
    assert json.loads((out / "fake-model.json").read_text()) == result
    if should_write:
        assert json.loads(measured.read_text()) == {
            "choiceAccuracy": 0.5,
            "machine": "test machine",
        }
    elif existing:
        assert measured.read_text() == original
    else:
        assert not measured.exists()


def test_model_size_includes_external_weight_files(tmp_path):
    from noulo.benchmark.run import model_size_bytes

    (tmp_path / "model.onnx").write_bytes(b"g" * 10)
    (tmp_path / "model_q4.onnx_data").write_bytes(b"w" * 90)
    (tmp_path / "tokenizer.json").write_bytes(b"t" * 5)
    assert model_size_bytes(tmp_path) == 100
    assert model_size_bytes(tmp_path / "missing") is None


def test_worker_result_is_found_among_noisy_runtime_output():
    from noulo.benchmark.run import RESULT_PREFIX, parse_worker_output

    noisy = (
        "CoreML: logits has unbounded dimension\n"
        f'{RESULT_PREFIX}{{"model": "m", "p50Ms": 7.5}}\n'
        "more runtime chatter\n"
    )
    assert parse_worker_output(noisy) == {"model": "m", "p50Ms": 7.5}


def test_worker_output_without_result_is_reported_as_error():
    from noulo.benchmark.run import parse_worker_output

    assert "no result" in parse_worker_output("just warnings\n")["error"]


def test_worker_result_is_found_even_when_runtime_output_has_no_trailing_newline():
    from noulo.benchmark.run import RESULT_PREFIX, parse_worker_output

    glued = f'...dimension which is not supported..{RESULT_PREFIX}{{"model": "m"}}\n'
    assert parse_worker_output(glued) == {"model": "m"}
