from src.eval.metrics import compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_prob = [0.0, 0.1, 0.9, 1.0]
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["auc_roc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["npv"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_compute_metrics_all_wrong():
    y_true = [0, 1]
    y_prob = [0.9, 0.1]
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 0.0
    assert metrics["specificity"] == 0.0
    assert metrics["npv"] == 0.0


def test_compute_metrics_specificity_and_npv_on_mixed_predictions():
    # 1 true positive, 1 false positive, 1 true negative, 1 false negative
    y_true = [1, 0, 0, 1]
    y_prob = [0.9, 0.8, 0.1, 0.2]
    metrics = compute_metrics(y_true, y_prob)
    # tn=1 (idx2), fp=1 (idx1), tp=1 (idx0), fn=1 (idx3)
    assert metrics["specificity"] == 0.5  # tn / (tn+fp) = 1/2
    assert metrics["npv"] == 0.5  # tn / (tn+fn) = 1/2


def test_compute_metrics_threshold_affects_predictions():
    y_true = [0, 1]
    y_prob = [0.4, 0.6]
    strict = compute_metrics(y_true, y_prob, threshold=0.7)
    lenient = compute_metrics(y_true, y_prob, threshold=0.3)
    assert strict["recall"] == 0.0
    assert lenient["recall"] == 1.0
