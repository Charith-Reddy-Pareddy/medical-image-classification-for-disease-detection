import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Standard binary classification metrics. Recall/sensitivity matters
    most here -- a false negative is a missed diagnosis -- but all of
    accuracy/precision/recall/F1/AUC are reported per the project spec.

    AUC-ROC alone is misleading at low prevalence (NIH/OpenI pneumonia
    prevalence is a few percent): a model can hold a high AUC-ROC while
    precision collapses, because ROC-AUC doesn't see the class imbalance
    the way precision does. AUPRC (average_precision_score) is reported
    alongside it for exactly that reason, plus specificity/PPV/NPV/
    balanced accuracy so discrimination and threshold-dependent behavior
    aren't conflated into one number.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    has_both_classes = len(np.unique(y_true)) > 1

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),  # PPV
        "recall": recall_score(y_true, y_pred, zero_division=0),  # sensitivity
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "npv": tn / (tn + fn) if (tn + fn) > 0 else float("nan"),
        # like AUC/AUPRC, balanced accuracy (mean per-class recall) is
        # degenerate with only one true class present -- NaN rather than
        # a misleading number from sklearn's single-class fallback
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if has_both_classes else float("nan"),
        "auc_roc": roc_auc_score(y_true, y_prob) if has_both_classes else float("nan"),
        "auprc": average_precision_score(y_true, y_prob) if has_both_classes else float("nan"),
    }
    return metrics
