from src.eval.metrics import compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_prob = [0.0, 0.1, 0.9, 1.0]
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["auc_roc"] == 1.0


def test_compute_metrics_all_wrong():
    y_true = [0, 1]
    y_prob = [0.9, 0.1]
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 0.0


def test_compute_metrics_threshold_affects_predictions():
    y_true = [0, 1]
    y_prob = [0.4, 0.6]
    strict = compute_metrics(y_true, y_prob, threshold=0.7)
    lenient = compute_metrics(y_true, y_prob, threshold=0.3)
    assert strict["recall"] == 0.0
    assert lenient["recall"] == 1.0
