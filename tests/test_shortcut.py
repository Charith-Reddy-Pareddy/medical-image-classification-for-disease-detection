import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.interpret import shortcut
from src.interpret.shortcut import (
    border_mask_accuracy_drop,
    gradcam_overlap_fraction,
    per_image_overlaps,
    per_image_overlaps_multi_threshold,
)
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


class _ScriptedModel(torch.nn.Module):
    """Returns a fixed high/low logit per call, in order -- lets a test
    control which images are "correctly classified" deterministically
    without training a real model.
    """

    def __init__(self, predictions: list):
        super().__init__()
        self._logits = iter(10.0 if p == 1 else -10.0 for p in predictions)

    def forward(self, x):
        logit = next(self._logits)
        return torch.full((x.shape[0], 1), logit)


def _identity_transform(image):
    arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)


def test_per_image_overlaps_multi_threshold_matches_single_threshold_calls(tmp_path, monkeypatch):
    # 3 rows: predictions [1, 0, 0] vs labels [1, 0, 1] -- row 2 is a
    # misclassification and must be excluded from every threshold's output
    rows = []
    for i, label in enumerate([1, 0, 1]):
        path = tmp_path / f"img_{i}.png"
        Image.new("RGB", (10, 10)).save(path)
        rows.append({"path": str(path), "label": label})
    manifest_df = pd.DataFrame(rows)
    model = _ScriptedModel(predictions=[1, 0, 0])

    # fixed cam: top half "hot" (1.0), bottom half cold (0.0); fixed mask:
    # right half True -- overlap at threshold 0.5 is the top-right quadrant
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[:5, :] = 1.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, 5:] = True

    monkeypatch.setattr(shortcut, "compute_gradcam", lambda model, tensor, category: cam)
    monkeypatch.setattr(shortcut, "lung_mask", lambda path: mask)

    thresholds = [0.0, 0.5, 1.5]
    by_threshold = per_image_overlaps_multi_threshold(
        model, manifest_df, torch.device("cpu"), _identity_transform, thresholds
    )

    # only the 2 correctly classified rows survive, for every threshold
    assert set(by_threshold.keys()) == set(thresholds)
    for t in thresholds:
        assert len(by_threshold[t]) == 2

    assert all(r["overlap"] == 0.5 for r in by_threshold[0.0])
    assert all(r["overlap"] == 0.5 for r in by_threshold[0.5])
    assert all(r["overlap"] == 0.0 for r in by_threshold[1.5])  # no pixels clear this threshold

    # per_image_overlaps(cam_threshold=0.5) must agree with the
    # multi-threshold sweep's 0.5 entry (same underlying cam/mask)
    model_single = _ScriptedModel(predictions=[1, 0, 0])
    single = per_image_overlaps(
        model_single, manifest_df, torch.device("cpu"), _identity_transform, cam_threshold=0.5
    )
    assert [r["overlap"] for r in single] == [r["overlap"] for r in by_threshold[0.5]]


def test_border_mask_accuracy_drop_runs_and_returns_expected_keys():
    model = BaselineCNN()
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 2, (8,)).float()
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    result = border_mask_accuracy_drop(model, loader, device=torch.device("cpu"), border_frac=0.1)

    assert set(result.keys()) == {"accuracy_original", "accuracy_masked", "accuracy_drop"}
    assert 0.0 <= result["accuracy_original"] <= 1.0
    assert 0.0 <= result["accuracy_masked"] <= 1.0
