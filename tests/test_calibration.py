import pytest

from src.eval.calibration import brier_score, expected_calibration_error, reliability_curve


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score([0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0]) == 0.0


def test_brier_score_worst_case_is_one():
    assert brier_score([0, 1], [1.0, 0.0]) == 1.0


def test_brier_score_uninformative_half_on_balanced_set():
    assert brier_score([0, 1], [0.5, 0.5]) == pytest.approx(0.25)


def test_reliability_curve_bins_and_rates():
    y_true = [0, 0, 1, 1]
    y_prob = [0.02, 0.08, 0.92, 0.98]
    bins = reliability_curve(y_true, y_prob, n_bins=10)
    # low-probability bin: both true negatives -> observed rate 0
    low_bin = next(b for b in bins if b["bin_lower"] == 0.0)
    assert low_bin["observed_rate"] == 0.0
    assert low_bin["n"] == 2
    # high-probability bin: both true positives -> observed rate 1
    high_bin = next(b for b in bins if b["bin_upper"] == 1.0)
    assert high_bin["observed_rate"] == 1.0
    assert high_bin["n"] == 2


def test_reliability_curve_omits_empty_bins():
    y_true = [0, 1]
    y_prob = [0.05, 0.95]
    bins = reliability_curve(y_true, y_prob, n_bins=10)
    assert len(bins) == 2


def test_ece_perfect_calibration_is_zero():
    # p=0.0 bin is all-negative (observed rate 0), p=1.0 bin is all-positive
    # (observed rate 1) -- predicted probability matches observed exactly
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_prob = [0.0] * 4 + [1.0] * 6
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_miscalibration():
    # model says 0.9 confident but is only right half the time
    y_true = [0, 1] * 10
    y_prob = [0.9] * 20
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.4, abs=1e-9)
