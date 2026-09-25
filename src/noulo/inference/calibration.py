"""Calibration layer: maps a raw model probability to a calibrated probability.

Kept separate from the base model so a new calibration file can be dropped in
without replacing the model. Every calibrator is a pure function p -> p' with
0 <= p' <= 1, and serialises to/from a small JSON document.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

_EPS = 1e-7


def _check(p: float) -> float:
    p = float(p)
    if math.isnan(p):
        raise ValueError("Cannot calibrate NaN.")
    return min(1.0, max(0.0, p))


def _logit(p: float) -> float:
    p = min(1.0 - _EPS, max(_EPS, p))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class Calibrator(Protocol):
    def __call__(self, p: float) -> float: ...

    def to_dict(self) -> dict: ...


@dataclass(frozen=True)
class IdentityCalibrator:
    def __call__(self, p: float) -> float:
        return _check(p)

    def to_dict(self) -> dict:
        return {"method": "identity"}


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float

    def __post_init__(self):
        if not self.temperature > 0:
            raise ValueError("Temperature must be positive.")

    def __call__(self, p: float) -> float:
        return _check(_sigmoid(_logit(_check(p)) / self.temperature))

    def to_dict(self) -> dict:
        return {"method": "temperature", "temperature": self.temperature}


@dataclass(frozen=True)
class PlattCalibrator:
    a: float
    b: float

    def __call__(self, p: float) -> float:
        return _check(_sigmoid(self.a * _logit(_check(p)) + self.b))

    def to_dict(self) -> dict:
        return {"method": "platt", "a": self.a, "b": self.b}


@dataclass(frozen=True)
class IsotonicCalibrator:
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    def __post_init__(self):
        if len(self.x) != len(self.y) or not self.x:
            raise ValueError("Isotonic knots must be non-empty and of equal length.")
        if any(b < a for a, b in zip(self.x, self.x[1:])):
            raise ValueError("Isotonic x knots must be non-decreasing.")
        if any(b < a for a, b in zip(self.y, self.y[1:])):
            raise ValueError("Isotonic y knots must be non-decreasing.")

    def __call__(self, p: float) -> float:
        return _check(float(np.interp(_check(p), self.x, self.y)))

    def to_dict(self) -> dict:
        return {"method": "isotonic", "x": list(self.x), "y": list(self.y)}


def calibrator_from_dict(data: dict) -> Calibrator:
    method = data.get("method")
    if method == "identity":
        return IdentityCalibrator()
    if method == "temperature":
        return TemperatureCalibrator(temperature=float(data["temperature"]))
    if method == "platt":
        return PlattCalibrator(a=float(data["a"]), b=float(data["b"]))
    if method == "isotonic":
        return IsotonicCalibrator(x=[float(v) for v in data["x"]], y=[float(v) for v in data["y"]])
    raise ValueError(f"Unknown calibration method: {method!r}")


def load_calibrator(path: str | Path) -> Calibrator:
    """Load a calibration file; a missing file means 'uncalibrated' (identity)."""
    path = Path(path)
    if not path.exists():
        return IdentityCalibrator()
    return calibrator_from_dict(json.loads(path.read_text()))


# --------------------------------------------------------------------------- fitting


def _nll(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.clip(pred, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))


def _logits(raw: np.ndarray) -> np.ndarray:
    raw = np.clip(raw, _EPS, 1 - _EPS)
    return np.log(raw / (1 - raw))


def _fit_temperature(raw: np.ndarray, y: np.ndarray) -> TemperatureCalibrator:
    z = _logits(raw)

    def loss(log_t: float) -> float:
        return _nll(1 / (1 + np.exp(-z / math.exp(log_t))), y)

    # Golden-section search over log(T) in [-3, 3]; the NLL is unimodal in T.
    lo, hi = -3.0, 3.0
    ratio = (math.sqrt(5) - 1) / 2
    c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    for _ in range(80):
        if loss(c) < loss(d):
            hi = d
        else:
            lo = c
        c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    return TemperatureCalibrator(temperature=math.exp((lo + hi) / 2))


def _fit_platt(raw: np.ndarray, y: np.ndarray) -> PlattCalibrator:
    """Logistic regression on the raw logit: damped Newton with backtracking."""
    x = np.column_stack([_logits(raw), np.ones_like(raw)])

    def predict(weights: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x @ weights, -500, 500)))

    w = np.array([1.0, 0.0])
    current = _nll(predict(w), y)
    for _ in range(100):
        p = predict(w)
        grad = x.T @ (p - y)
        hess = x.T @ (x * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        scale = 1.0
        while scale > 1e-6:
            candidate = w - scale * step
            loss = _nll(predict(candidate), y)
            if loss <= current:
                break
            scale /= 2
        else:
            break
        w, improvement, current = candidate, current - loss, loss
        if improvement < 1e-12:
            break
    return PlattCalibrator(a=float(w[0]), b=float(w[1]))


def _fit_isotonic(raw: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
    """Pool-adjacent-violators over raw scores (ties pre-aggregated)."""
    xs, inverse = np.unique(raw, return_inverse=True)
    sums = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse).astype(float)
    blocks: list[list[float]] = []  # [sum_y, count, sum_x*count]
    for x_val, s, n in zip(xs, sums, counts):
        blocks.append([s, n, x_val * n])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, n2, sx2 = blocks.pop()
            blocks[-1][0] += s2
            blocks[-1][1] += n2
            blocks[-1][2] += sx2
    knots_x = [b[2] / b[1] for b in blocks]
    knots_y = [b[0] / b[1] for b in blocks]
    return IsotonicCalibrator(x=knots_x, y=knots_y)


def fit_calibrator(method: str, raw_scores, labels) -> Calibrator:
    raw = np.asarray(raw_scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if raw.shape != y.shape or raw.size == 0:
        raise ValueError("raw_scores and labels must be non-empty and the same length.")
    if method == "identity":
        return IdentityCalibrator()
    if method == "temperature":
        return _fit_temperature(raw, y)
    if method == "platt":
        return _fit_platt(raw, y)
    if method == "isotonic":
        return _fit_isotonic(raw, y)
    raise ValueError(f"Unknown calibration method: {method!r}")
