import json
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from noulo.inference.calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    calibrator_from_dict,
    fit_calibrator,
    load_calibrator,
)

probabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def _all_calibrators():
    return [
        IdentityCalibrator(),
        TemperatureCalibrator(temperature=2.0),
        TemperatureCalibrator(temperature=0.5),
        PlattCalibrator(a=1.7, b=-0.3),
        IsotonicCalibrator(x=[0.0, 0.3, 0.7, 1.0], y=[0.05, 0.2, 0.9, 0.97]),
    ]


def test_identity_returns_input():
    assert IdentityCalibrator()(0.73) == pytest.approx(0.73)


def test_temperature_keeps_midpoint_at_half():
    assert TemperatureCalibrator(temperature=3.0)(0.5) == pytest.approx(0.5)


def test_temperature_above_one_softens_confidence():
    assert TemperatureCalibrator(temperature=2.0)(0.95) < 0.95
    assert TemperatureCalibrator(temperature=2.0)(0.05) > 0.05


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        TemperatureCalibrator(temperature=0.0)


def test_platt_applies_sigmoid_of_affine_logit():
    p = 0.8
    expected = 1 / (1 + math.exp(-(2.0 * math.log(p / (1 - p)) + 0.5)))
    assert PlattCalibrator(a=2.0, b=0.5)(p) == pytest.approx(expected)


def test_isotonic_interpolates_between_knots():
    cal = IsotonicCalibrator(x=[0.0, 1.0], y=[0.2, 0.8])
    assert cal(0.5) == pytest.approx(0.5)
    assert cal(0.0) == pytest.approx(0.2)


def test_isotonic_rejects_non_monotonic_knots():
    with pytest.raises(ValueError):
        IsotonicCalibrator(x=[0.0, 0.5, 1.0], y=[0.1, 0.9, 0.4])


@pytest.mark.parametrize("calibrator", _all_calibrators(), ids=lambda c: type(c).__name__)
@given(p=probabilities)
def test_every_calibrator_output_stays_in_unit_interval(calibrator, p):
    value = calibrator(p)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("calibrator", _all_calibrators(), ids=lambda c: type(c).__name__)
def test_every_calibrator_rejects_nan(calibrator):
    with pytest.raises(ValueError):
        calibrator(float("nan"))


@pytest.mark.parametrize("calibrator", _all_calibrators(), ids=lambda c: type(c).__name__)
def test_calibrator_roundtrips_through_dict(calibrator):
    restored = calibrator_from_dict(calibrator.to_dict())
    for p in (0.0, 0.1, 0.5, 0.85, 1.0):
        assert restored(p) == pytest.approx(calibrator(p))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibrator_from_dict({"method": "magic"})


def test_load_calibrator_reads_json_file(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"method": "platt", "a": 1.0, "b": 0.0}))
    assert isinstance(load_calibrator(path), PlattCalibrator)


def test_load_calibrator_missing_file_falls_back_to_identity(tmp_path):
    assert isinstance(load_calibrator(tmp_path / "absent.json"), IdentityCalibrator)


def _overconfident_dataset(n=400, seed=0):
    """Scores that are far more extreme than the true outcome rate."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(0, 1, n) < true_p).astype(float)
    logits = np.log(true_p / (1 - true_p)) * 4.0
    raw = 1 / (1 + np.exp(-logits))
    return raw, y


def _nll(pred, y):
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic"])
def test_fitting_improves_log_loss_on_overconfident_scores(method):
    raw, y = _overconfident_dataset()
    cal = fit_calibrator(method, raw, y)
    calibrated = np.array([cal(p) for p in raw])
    assert _nll(calibrated, y) < _nll(raw, y)


def test_fitted_temperature_softens_overconfident_scores():
    raw, y = _overconfident_dataset()
    cal = fit_calibrator("temperature", raw, y)
    assert cal.temperature > 1.5


def test_fitted_isotonic_is_monotonic():
    raw, y = _overconfident_dataset()
    cal = fit_calibrator("isotonic", raw, y)
    grid = np.linspace(0, 1, 101)
    values = [cal(p) for p in grid]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
