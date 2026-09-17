"""Calibration diagnostics, separate from discrimination (AUC-ROC/AUPRC).

A model can discriminate well (rank pneumonia above normal) while being
badly calibrated (its 0.9 doesn't mean "90% likely") -- the report's
observation that external accuracy partly reflects prevalence/threshold
mismatch, not just a discrimination failure, is checked quantitatively
here rather than asserted.
"""

import numpy as np


def brier_score(y_true, y_prob) -> float:
    """Mean squared error between predicted probability and outcome.
    Lower is better; 0 is perfect, 0.25 is what an uninformative p=0.5
    constant predictor scores on a balanced set.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_curve(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    """Bins predictions by predicted probability and reports, per bin,
    the mean predicted probability vs. the observed positive rate --
    the data behind a reliability diagram. Empty bins are omitted.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bins.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_predicted": float(y_prob[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    return bins


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE: the weighted-average gap between predicted probability and
    observed frequency across bins, weighted by bin size. 0 is perfect
    calibration.
    """
    n = len(np.asarray(y_true))
    bins = reliability_curve(y_true, y_prob, n_bins)
    if not bins or n == 0:
        return float("nan")
    return float(sum(b["n"] / n * abs(b["mean_predicted"] - b["observed_rate"]) for b in bins))
