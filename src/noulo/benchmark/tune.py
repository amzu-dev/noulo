"""Per-model tuning on the *calibration* split only (the test split stays held out).

- Choice / Score: pick the hypothesis template, premise mode, scoring method
  (and Score softmax temperature) that minimise error on the calibration split.
- Noul: choose a calibration method by k-fold cross-validated log loss, then
  fit it on the whole calibration split.

Outputs `profile.json` (NliBackend kwargs) and `calibration.json` in the model dir.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..inference import choice as choice_mod
from ..inference import score as score_mod
from ..inference.calibration import fit_calibrator
from ..inference.nli_backend import METHODS, PREMISES, NliBackend
from ..inference.score import expected_score
from .data import DEFAULT_DATA_DIR, load_dataset
from .metrics import log_loss

CALIBRATION_METHODS = ("identity", "temperature", "platt", "isotonic")
SCORE_TEMPERATURES = (0.5, 1.0, 2.0, 4.0)


def best_strategy(losses: dict[str, float]) -> str:
    """Lowest loss wins; ties keep the earliest candidate."""
    return min(losses, key=lambda k: (losses[k], list(losses).index(k)))


def choose_calibration_method(
    raw: Sequence[float], labels: Sequence[float], folds: int = 5, seed: int = 0
) -> tuple[str, dict[str, float]]:
    raw, labels = np.asarray(raw, float), np.asarray(labels, float)
    order = np.random.default_rng(seed).permutation(raw.size)
    chunks = np.array_split(order, folds)
    scores: dict[str, float] = {}
    for method in CALIBRATION_METHODS:
        losses = []
        for i, held_out in enumerate(chunks):
            train = np.concatenate([c for j, c in enumerate(chunks) if j != i])
            calibrator = fit_calibrator(method, raw[train], labels[train])
            losses.append(log_loss([calibrator(p) for p in raw[held_out]], labels[held_out]))
        scores[method] = float(np.mean(losses))
    return best_strategy(scores), scores


class CachingModel:
    """Memoises predict_logits per (premise, hypothesis) pair across strategy trials."""

    def __init__(self, model: Any):
        self._model, self.labels, self._cache = model, model.labels, {}

    def predict_logits(self, pairs):
        missing = [p for p in dict.fromkeys(pairs) if p not in self._cache]
        if missing:
            for pair, row in zip(missing, self._model.predict_logits(missing)):
                self._cache[pair] = row
        return np.array([self._cache[p] for p in pairs], dtype=np.float32).reshape(len(pairs), -1)

    def close(self):
        pass


def _choice_error(backend: NliBackend, rows: list[dict]) -> float:
    wrong = 0
    for r in rows:
        probs = backend.choice(r["input"], r["question"], [c["text"] for c in r["choices"]])
        wrong += r["choices"][int(np.argmax(probs))]["id"] != r["answer"]
    return wrong / len(rows)


def _score_error(backend: NliBackend, rows: list[dict]) -> float:
    errors = []
    for r in rows:
        value = expected_score(backend.score(r["input"], r["question"], r["rubric"]))
        errors.append(abs(value - r["expected_level"] / (len(r["rubric"]) - 1)))
    return float(np.mean(errors))


def tune_nli_model(
    model: Any,
    model_id: str,
    quantization: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    progress: Callable[[str], None] = lambda _msg: None,
) -> dict[str, Any]:
    cached = CachingModel(model)
    choice_rows = load_dataset(data_dir / "choice.jsonl", split="calibration")
    score_rows = load_dataset(data_dir / "score.jsonl", split="calibration")

    choice_losses: dict[str, float] = {}
    choice_grid: dict[str, dict[str, Any]] = {}
    for template, premise, method in itertools.product(choice_mod.TEMPLATES, PREMISES, METHODS):
        kwargs = {"choice_template": template, "choice_premise": premise, "choice_method": method}
        key = json.dumps(kwargs, sort_keys=True)
        backend = NliBackend(cached, model_id=model_id, quantization=quantization, **kwargs)
        choice_losses[key], choice_grid[key] = _choice_error(backend, choice_rows), kwargs
    best_choice = best_strategy(choice_losses)
    progress(f"choice: best {choice_grid[best_choice]} error={choice_losses[best_choice]:.3f}")

    score_losses: dict[str, float] = {}
    score_grid: dict[str, dict[str, Any]] = {}
    for template, premise, method, temp in itertools.product(
        score_mod.TEMPLATES, PREMISES, METHODS, SCORE_TEMPERATURES
    ):
        kwargs = {
            "score_template": template,
            "score_premise": premise,
            "score_method": method,
            "score_temperature": temp,
        }
        key = json.dumps(kwargs, sort_keys=True)
        backend = NliBackend(cached, model_id=model_id, quantization=quantization, **kwargs)
        score_losses[key], score_grid[key] = _score_error(backend, score_rows), kwargs
    best_score = best_strategy(score_losses)
    progress(f"score: best {score_grid[best_score]} MAE={score_losses[best_score]:.3f}")

    return {
        "profile": {**choice_grid[best_choice], **score_grid[best_score]},
        "choiceError": choice_losses[best_choice],
        "scoreMae": score_losses[best_score],
    }


def tune_noul_calibration(
    noul: Callable[[str, str], float],
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    rows = [
        r
        for r in load_dataset(data_dir / "noul.jsonl", split="calibration")
        if r["label"] in ("yes", "no")
    ]
    raw = np.array([noul(r["input"], r["proposition"]) for r in rows])
    labels = np.array([1.0 if r["label"] == "yes" else 0.0 for r in rows])
    method, scores = choose_calibration_method(raw, labels)
    calibrator = fit_calibrator(method, raw, labels)
    return {"calibration": calibrator.to_dict(), "method": method, "cvLogLoss": scores}


def write_tuning(
    model_dir: Path, profile: dict[str, Any] | None, calibration: dict[str, Any]
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    if profile is not None:
        (model_dir / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    (model_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
