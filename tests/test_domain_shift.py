import torch
from torch.utils.data import DataLoader, TensorDataset

from src.eval.domain_shift import evaluate_across_datasets
from src.models.baseline_cnn import BaselineCNN


def _make_loader(n, label_value):
    images = torch.randn(n, 3, 32, 32)
    labels = torch.full((n,), label_value, dtype=torch.float32)
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def test_evaluate_across_datasets_reports_metrics_per_dataset():
    model = BaselineCNN()
    loaders = {
        "site_a": _make_loader(8, 1.0),
        "site_b": _make_loader(8, 0.0),
    }
    results = evaluate_across_datasets(model, loaders, device=torch.device("cpu"))

    assert set(results.keys()) == {"site_a", "site_b"}
    for name, result in results.items():
        assert len(result["y_true"]) == 8
        assert len(result["y_prob"]) == 8
        assert "accuracy" in result["metrics"]
