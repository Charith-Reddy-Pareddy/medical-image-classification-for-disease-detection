import numpy as np
from sklearn.metrics import roc_auc_score

from src.eval.stats import bootstrap_ci, mcnemar_test, sensitivity_at_threshold


def test_bootstrap_ci_perfect_separation_is_near_one():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([rng.uniform(0, 0.3, 50), rng.uniform(0.7, 1.0, 50)])

    point, lower, upper = bootstrap_ci(y_true, y_prob, roc_auc_score, n_boot=200)
    assert point == 1.0
    assert lower <= point <= upper
    assert lower > 0.9


def test_bootstrap_ci_random_predictions_interval_contains_half():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 300)
    y_prob = rng.uniform(0, 1, 300)

    point, lower, upper = bootstrap_ci(y_true, y_prob, roc_auc_score, n_boot=300)
    assert lower < 0.5 < upper


def test_sensitivity_at_threshold():
    fn = sensitivity_at_threshold(threshold=0.5)
    y_true = [0, 1, 1, 1]
    y_prob = [0.9, 0.9, 0.1, 0.6]
    # 2 of 3 positives correctly flagged
    assert fn(y_true, y_prob) == 2 / 3


def test_mcnemar_identical_predictions_gives_p_one():
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    result = mcnemar_test(y_true, y_pred, y_pred)
    assert result["p_value"] == 1.0
    assert result["b"] == 0 and result["c"] == 0


def test_mcnemar_detects_strong_asymmetry():
    y_true = [1] * 20
    y_pred_a = [1] * 20  # always correct
    y_pred_b = [0] * 20  # always wrong
    result = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert result["b"] == 20
    assert result["c"] == 0
    assert result["p_value"] < 0.001
