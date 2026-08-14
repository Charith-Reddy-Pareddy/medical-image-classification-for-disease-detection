import pandas as pd
from PIL import Image

from src.data.dataset import ChestXrayDataset, get_transforms


def _make_manifest(tmp_path, n=4):
    rows = []
    for i in range(n):
        path = tmp_path / f"img_{i}.jpeg"
        Image.new("RGB", (32, 32), color=(i * 10, 0, 0)).save(path)
        rows.append({"path": str(path), "label": i % 2})
    return pd.DataFrame(rows)


def test_dataset_length(tmp_path):
    manifest = _make_manifest(tmp_path)
    ds = ChestXrayDataset(manifest)
    assert len(ds) == 4


def test_dataset_returns_transformed_tensor_and_int_label(tmp_path):
    manifest = _make_manifest(tmp_path)
    ds = ChestXrayDataset(manifest, transform=get_transforms(image_size=64, train=False))

    image, label = ds[0]
    assert image.shape == (3, 64, 64)
    assert isinstance(label, int)


def test_train_transform_includes_augmentation():
    train_tf = get_transforms(image_size=64, train=True)
    eval_tf = get_transforms(image_size=64, train=False)
    assert len(train_tf.transforms) > len(eval_tf.transforms)
