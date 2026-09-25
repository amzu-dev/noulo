"""Benchmark runner.

    noulo benchmark                         # active model
    noulo benchmark --model a --model b     # compare models
    noulo benchmark --all --download --tune --compare docs/model-comparison.md

Each model is measured in a fresh subprocess so peak RAM and cold start are
real, per-model numbers. Tuning (--tune) only ever looks at the calibration
split; all reported accuracy/calibration numbers come from the test split.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .data import DEFAULT_DATA_DIR, load_dataset
from .report import format_comparison, format_report

MEASURED_KEYS = (
    "modelSizeBytes",
    "peakRssBytes",
    "coldStartMs",
    "noulAccuracy",
    "noulEce",
    "choiceAccuracy",
    "scoreMae",
    "p50Ms",
    "p95Ms",
)


def machine_name() -> str:
    import platform

    if sys.platform == "darwin":
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if brand:
            return brand
    return platform.processor() or platform.machine()


def write_measured(model_dir: Path, result: dict[str, Any], *, machine: str) -> None:
    """Persist the numbers the model pickers show (size, RAM, accuracy, latency)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    data = {key: result[key] for key in MEASURED_KEYS if key in result}
    data["machine"] = machine
    (model_dir / "measured.json").write_text(json.dumps(data, indent=2) + "\n")


RESULT_PREFIX = "NOULO_RESULT:"  # runtimes (e.g. CoreML) may print to stdout too


def parse_worker_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        start = line.find(RESULT_PREFIX)
        if start >= 0:  # runtime chatter without a newline may precede the marker
            return json.loads(line[start + len(RESULT_PREFIX) :])
    return {"error": "worker printed no result"}


def model_size_bytes(model_dir: Path) -> int | None:
    """Size of the model weights: model.onnx plus any external *.onnx_data files."""
    graph = model_dir / "model.onnx"
    if not graph.exists():
        return None
    return graph.stat().st_size + sum(f.stat().st_size for f in model_dir.glob("*.onnx_data*"))


def _peak_rss_bytes() -> int:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:  # Windows
        import psutil

        return int(psutil.Process().memory_info().peak_wset)


def measure(
    model_id: str,
    models_dir: Path,
    models_file: Path | None,
    data_dir: Path,
    split: str = "test",
    low_memory: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load one model in this process and evaluate it; returns measured metrics."""
    t_process = time.perf_counter()
    from ..config import Settings
    from ..inference.engine import DecisionEngine
    from ..runtime import build_registry
    from .metrics import (
        accuracy,
        brier_score,
        expected_calibration_error,
        mean_absolute_error,
        percentile_ms,
    )

    settings = Settings(
        _env_file=None,
        models_dir=models_dir,
        models_file=models_file,
        low_memory=low_memory,
        device=device,
    )
    registry = build_registry(settings)
    engine = DecisionEngine(
        load_backend=registry.load_backend,
        model_id=model_id,
        load_calibrator=registry.load_calibrator,
    )
    t_start = time.perf_counter()
    engine.start()
    cold_start_ms = (time.perf_counter() - t_start) * 1000
    process_start_ms = (time.perf_counter() - t_process) * 1000
    info = engine.model_info

    from ..api.validation import ChoiceOption

    latencies: list[float] = []

    def timed(fn, *args, **kwargs):
        t = time.perf_counter()
        result = fn(*args, **kwargs)
        latencies.append(time.perf_counter() - t)
        return result

    noul_rows = load_dataset(data_dir / "noul.jsonl", split=split)
    binary = [r for r in noul_rows if r["label"] in ("yes", "no")]
    values, raws, labels = [], [], []
    for r in binary:
        result = timed(engine.noul, r["input"], r["proposition"], diagnostics=True)
        values.append(result.value)
        raws.append(result.diagnostics.get("raw", result.value))
        labels.append(1.0 if r["label"] == "yes" else 0.0)
    unknown = [
        timed(engine.noul, r["input"], r["proposition"]).value
        for r in noul_rows
        if r["label"] == "unknown"
    ]

    choice_rows = load_dataset(data_dir / "choice.jsonl", split=split)
    chosen = [
        timed(
            engine.choice, r["input"], r["question"], [ChoiceOption(**c) for c in r["choices"]]
        ).value
        for r in choice_rows
    ]

    score_rows = load_dataset(data_dir / "score.jsonl", split=split)
    scores = [timed(engine.score, r["input"], r["question"], r["rubric"]).value for r in score_rows]
    expected_scores = [r["expected_level"] / (len(r["rubric"]) - 1) for r in score_rows]

    engine.shutdown()
    size_bytes = model_size_bytes(models_dir / model_id)
    return {
        "model": info.id,
        "backend": info.backend,
        "device": info.device,
        "quantization": info.quantization,
        "modelSizeBytes": size_bytes,
        "peakRssBytes": _peak_rss_bytes(),
        "coldStartMs": cold_start_ms,
        "processStartMs": process_start_ms,
        "choiceAccuracy": accuracy(chosen, [r["answer"] for r in choice_rows]),
        "noulAccuracy": accuracy([v >= 0.5 for v in values], [bool(y) for y in labels]),
        "noulEce": expected_calibration_error(values, labels),
        "noulBrier": brier_score(values, labels),
        "noulRawEce": expected_calibration_error(raws, labels),
        "noulUnknownDeviation": (
            sum(abs(v - 0.5) for v in unknown) / len(unknown) if unknown else None
        ),
        "scoreMae": mean_absolute_error(scores, expected_scores),
        "p50Ms": percentile_ms(latencies, 50),
        "p95Ms": percentile_ms(latencies, 95),
        "throughputRps": len(latencies) / sum(latencies) if latencies else None,
        "items": {"noul": len(noul_rows), "choice": len(choice_rows), "score": len(score_rows)},
        "split": split,
    }


def _run_worker(model_id: str, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "noulo.benchmark.run",
        "--worker",
        model_id,
        "--models-dir",
        str(args.models_dir),
        "--data-dir",
        str(args.data_dir),
        "--split",
        args.split,
    ]
    if args.models_file:
        cmd += ["--models-file", str(args.models_file)]
    if args.low_memory:
        cmd.append("--low-memory")
    cmd += ["--device", args.device]
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        return {"model": model_id, "error": tail[:200]}
    result = parse_worker_output(proc.stdout)
    result.setdefault("model", model_id)
    return result


def _tune(model_id: str, args: argparse.Namespace) -> None:
    from ..config import Settings
    from ..runtime import build_registry
    from .tune import tune_nli_model, tune_noul_calibration, write_tuning

    registry = build_registry(
        Settings(_env_file=None, models_dir=args.models_dir, models_file=args.models_file)
    )
    model_dir = Path(args.models_dir) / model_id
    for stale in ("profile.json", "calibration.json"):  # tune from a clean slate
        (model_dir / stale).unlink(missing_ok=True)
    backend = registry.load_backend(model_id)
    profile = None
    if backend.info.backend == "onnx-nli":
        from ..inference.model import OnnxNliModel

        model = OnnxNliModel(model_dir)
        tuned = tune_nli_model(
            model,
            model_id,
            backend.info.quantization,
            args.data_dir,
            progress=lambda m: print(f"  [{model_id}] {m}"),
        )
        profile = tuned["profile"]
        model.close()
    calibration = tune_noul_calibration(backend.noul, args.data_dir)
    print(f"  [{model_id}] noul calibration: {calibration['method']}")
    backend.close()
    write_tuning(model_dir, profile, calibration["calibration"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="noulo benchmark")
    parser.add_argument("--model", action="append", help="model id (repeatable)")
    parser.add_argument("--all", action="store_true", help="all catalog NLI models")
    parser.add_argument("--download", action="store_true", help="download missing models first")
    parser.add_argument("--tune", action="store_true", help="tune profile/calibration first")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--models-file", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test", choices=["test", "calibration"])
    parser.add_argument("--low-memory", action="store_true", help="measure with NOULO_LOW_MEMORY")
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu (default, comparable numbers), auto, gpu, coreml, cuda, ...",
    )
    parser.add_argument("--out", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--compare", type=Path, help="write a markdown comparison table here")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        print(
            "\n"
            + RESULT_PREFIX
            + json.dumps(
                measure(
                    args.worker,
                    args.models_dir,
                    args.models_file,
                    args.data_dir,
                    args.split,
                    args.low_memory,
                    args.device,
                )
            )
        )
        return 0

    from ..config import Settings
    from ..registry import CATALOG
    from ..runtime import build_registry

    if args.all:
        model_ids = [e.id for e in CATALOG if e.kind == "nli"]
    else:
        model_ids = args.model or [Settings().model]
    registry = build_registry(
        Settings(_env_file=None, models_dir=args.models_dir, models_file=args.models_file)
    )
    installed = {m["id"] for m in registry.list() if m["installed"]}

    results = []
    for model_id in model_ids:
        if model_id not in installed:
            if not args.download:
                results.append({"model": model_id, "error": "not installed"})
                continue
            print(f"Downloading {model_id} ...")
            registry.download(model_id)
        if args.tune:
            print(f"Tuning {model_id} on the calibration split ...")
            _tune(model_id, args)
        print(f"Benchmarking {model_id} ...")
        result = _run_worker(model_id, args)
        results.append(result)
        if "error" in result:
            print(f"  failed: {result['error']}")
            continue
        print(format_report(result), end="\n\n")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{model_id}.json").write_text(json.dumps(result, indent=2) + "\n")
        if not args.low_memory and args.device == "cpu":  # menus show CPU baseline numbers
            write_measured(Path(args.models_dir) / model_id, result, machine=machine_name())

    if args.compare:
        args.compare.parent.mkdir(parents=True, exist_ok=True)
        args.compare.write_text(format_comparison(results) + "\n")
        print(f"Wrote {args.compare}")
    return 0 if all("error" not in r for r in results) else 1


def main_and_exit(argv: list[str] | None = None) -> None:
    """Run and exit immediately.

    After many ONNX Runtime sessions and worker subprocesses, native static
    destructors can race at interpreter exit on macOS ("recursive_mutex lock
    failed"). All results are written by then, so skip them.
    """
    code = main(argv)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main_and_exit()
