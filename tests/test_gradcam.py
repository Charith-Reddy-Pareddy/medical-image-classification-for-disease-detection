import torch

from src.interpret.gradcam import compute_gradcam, target_layers_for
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer import DenseNet121Transfer, ResNet50Transfer


def test_target_layers_resolve_for_every_registered_model():
    for cls in (BaselineCNN, ResNet50Transfer, DenseNet121Transfer):
        model = cls(pretrained=False) if cls is not BaselineCNN else cls()
        layers = target_layers_for(model)
        assert len(layers) == 1


def test_compute_gradcam_returns_heatmap_matching_input_resolution():
    model = BaselineCNN()
    model.eval()
    image = torch.randn(3, 64, 64)

    cam = compute_gradcam(model, image, category=1)

    assert cam.shape == (64, 64)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
