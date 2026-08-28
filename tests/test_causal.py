import numpy as np
from PIL import Image

from src.interpret.causal import (
    build_causal_dataset,
    detect_text_marker,
    fit_shortcut_logistic_regression,
    image_resolution_features,
)


def test_image_resolution_features(tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (400, 200)).save(path)

    features = image_resolution_features(str(path))

    assert features["width"] == 400
    assert features["height"] == 200
    assert features["aspect_ratio"] == 2.0


def test_detect_text_marker_true_on_bright_corner_blobs(tmp_path):
    arr = np.full((300, 300), 50, dtype=np.uint8)
    # a few small, tightly packed bright blobs in one corner
    arr[5:8, 5:8] = 255
    arr[5:8, 15:18] = 255
    arr[5:8, 25:28] = 255
    path = tmp_path / "marked.png"
    Image.fromarray(arr).save(path)

    assert detect_text_marker(str(path)) is True


def test_detect_text_marker_false_on_uniform_image(tmp_path):
    arr = np.full((300, 300), 50, dtype=np.uint8)
    path = tmp_path / "plain.png"
    Image.fromarray(arr).save(path)

    assert detect_text_marker(str(path)) is False


def test_build_causal_dataset_and_fit_logistic_regression(tmp_path):
    rng = np.random.default_rng(0)
    records = []
    for i in range(200):
        age_group = i % 2  # alternate pediatric/adult
        path = tmp_path / f"img_{i}.png"
        # resolution is independent of age_group here -- only "overlap"
        # depends on it -- so the fitted age_group coefficient isn't
        # confounded by a built-in resolution correlation
        w = int(rng.integers(250, 650))
        h = int(rng.integers(250, 650))
        arr = np.full((h, w), int(rng.integers(0, 200)), dtype=np.uint8)
        if rng.random() < 0.4:
            arr[5:8, 5:8] = 255
            arr[5:8, 15:18] = 255
        Image.fromarray(arr).save(path)
        # adult group only *biased* toward low overlap -- distributions
        # overlap, so age_group doesn't perfectly separate the outcome
        overlap = float(np.clip(rng.normal(0.2 if age_group == 1 else 0.5, 0.15), 0.0, 1.0))
        records.append({"path": str(path), "overlap": overlap, "age_group": age_group})

    causal_df = build_causal_dataset(records)
    assert len(causal_df) == 200
    assert set(causal_df["age_group"].unique()) == {0, 1}
    assert set(causal_df["shortcut_driven"].unique()) <= {0, 1}

    result = fit_shortcut_logistic_regression(causal_df)
    assert "age_group" in result.params.index
    # the synthetic data was constructed so age_group predicts shortcut_driven
    assert result.params["age_group"] > 0
