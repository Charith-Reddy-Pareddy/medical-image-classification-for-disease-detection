import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.interpret.shortcut import border_mask_accuracy_drop, gradcam_overlap_fraction
from src.models.baseline_cnn import BaselineCNN


def test_gradcam_overlap_fraction_full_overlap():
    cam = np.ones((10, 10))
    mask = np.ones((10, 10), dtype=bool)
    assert gradcam_overlap_fraction(cam, mask, cam_threshold=0.5) == 1.0


def test_gradcam_overlap_fraction_no_overlap():
    cam = np.ones((10, 10))
    mask = np.zeros((10, 10), dtype=bool)
    assert gradcam_overlap_fraction(cam, mask, cam_threshold=0.5) == 0.0


def test_gradcam_overlap_fraction_no_hot_pixels_is_zero():
    cam = np.zeros((10, 10))
    mask = np.ones((10, 10), dtype=bool)
    assert gradcam_overlap_fraction(cam, mask, cam_threshold=0.5) == 0.0


def test_border_mask_accuracy_drop_runs_and_returns_expected_keys():
    model = BaselineCNN()
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 2, (8,)).float()
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    result = border_mask_accuracy_drop(model, loader, device=torch.device("cpu"), border_frac=0.1)

    assert set(result.keys()) == {"accuracy_original", "accuracy_masked", "accuracy_drop"}
    assert 0.0 <= result["accuracy_original"] <= 1.0
    assert 0.0 <= result["accuracy_masked"] <= 1.0
