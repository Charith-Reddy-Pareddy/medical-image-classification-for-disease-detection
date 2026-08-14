import torch

from src.models import MODEL_REGISTRY
from src.models.baseline_cnn import BaselineCNN


def test_baseline_cnn_output_shape():
    model = BaselineCNN()
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (4, 1)


def test_baseline_cnn_is_trainable_with_one_step():
    model = BaselineCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0.0, 1.0])

    optimizer.zero_grad()
    loss = criterion(model(x).squeeze(1), y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_model_registry_contains_baseline_cnn():
    assert MODEL_REGISTRY["baseline_cnn"] is BaselineCNN
