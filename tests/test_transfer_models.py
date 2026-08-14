import torch

from src.models import MODEL_REGISTRY
from src.models.transfer import DenseNet121Transfer, ResNet50Transfer


def test_resnet50_output_shape():
    model = ResNet50Transfer(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    assert model(x).shape == (2, 1)


def test_densenet121_output_shape():
    model = DenseNet121Transfer(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    assert model(x).shape == (2, 1)


def test_transfer_models_are_trainable_with_one_step():
    for cls in (ResNet50Transfer, DenseNet121Transfer):
        model = cls(pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()

        x = torch.randn(2, 3, 224, 224)
        y = torch.tensor([0.0, 1.0])

        optimizer.zero_grad()
        loss = criterion(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)


def test_registry_includes_transfer_models():
    assert MODEL_REGISTRY["resnet50"] is ResNet50Transfer
    assert MODEL_REGISTRY["densenet121"] is DenseNet121Transfer
