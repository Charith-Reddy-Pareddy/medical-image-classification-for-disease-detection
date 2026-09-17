import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    """Bootstrapped confidence interval for a metric_fn(y_true, y_prob) -> float,
    e.g. roc_auc_score or a sensitivity function. Resamples with replacement
    at the example level, n_boot times, and takes the percentile interval.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point_estimate = metric_fn(y_true, y_prob)

    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_true_bs, y_prob_bs = y_true[idx], y_prob[idx]
        if len(np.unique(y_true_bs)) < 2:
            continue  # metric undefined for a single-class resample
        scores.append(metric_fn(y_true_bs, y_prob_bs))

    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return point_estimate, lower, upper


def bootstrap_ci_clustered(
    y_true, y_prob, groups, metric_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42
):
    """Patient/study-level bootstrapped CI: resamples whole groups (e.g.
    patient IDs) with replacement rather than individual examples, so
    multiple images from the same patient move together. Plain
    example-level `bootstrap_ci` treats those images as independent
    observations, which understates the true uncertainty whenever a
    dataset has more than one image per patient.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    rng = np.random.default_rng(seed)

    point_estimate = metric_fn(y_true, y_prob)

    group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    scores = []
    for _ in range(n_boot):
        sampled_groups = unique_groups[rng.integers(0, n_groups, size=n_groups)]
        idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        y_true_bs, y_prob_bs = y_true[idx], y_prob[idx]
        if len(np.unique(y_true_bs)) < 2:
            continue  # metric undefined for a single-class resample
        scores.append(metric_fn(y_true_bs, y_prob_bs))

    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return point_estimate, lower, upper


def sensitivity_at_threshold(threshold: float = 0.5):
    """Returns a metric_fn(y_true, y_prob) usable with bootstrap_ci."""

    def _fn(y_true, y_prob):
        y_true = np.asarray(y_true)
        y_pred = (np.asarray(y_prob) >= threshold).astype(int)
        positives = y_true == 1
        if positives.sum() == 0:
            return float("nan")
        return (y_pred[positives] == 1).mean()

    return _fn


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """Exact McNemar's test for whether two models' error rates differ on
    the same test set. Compares only the discordant pairs: cases where one
    model got it right and the other didn't.
    """
    y_true = np.asarray(y_true)
    correct_a = np.asarray(y_pred_a) == y_true
    correct_b = np.asarray(y_pred_b) == y_true

    # b: A right, B wrong. c: A wrong, B right.
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return {"b": b, "c": c, "p_value": 1.0}

    p_value = binomtest(min(b, c), b + c, 0.5).pvalue
    return {"b": b, "c": c, "p_value": p_value}
