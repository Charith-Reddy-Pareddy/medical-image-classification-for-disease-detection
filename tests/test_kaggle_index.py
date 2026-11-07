import pytest

from chest_xray_audit.data_manifest import load_dataset_manifest
from chest_xray_audit.kaggle_index import DatasetLayoutError, build_kaggle_index


def _make_kaggle_tree(root):
    for split in ("train", "val", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            (root / "data/kaggle/chest_xray" / split / label).mkdir(parents=True)


def test_builds_patient_grouped_kaggle_records(tmp_path):
    _make_kaggle_tree(tmp_path)
    normal = tmp_path / "data/kaggle/chest_xray/train/NORMAL/person12_1.jpeg"
    pneumonia = tmp_path / "data/kaggle/chest_xray/test/PNEUMONIA/person34_bacteria_2.PNG"
    normal.touch()
    pneumonia.touch()

    records = build_kaggle_index(load_dataset_manifest(), project_root=tmp_path)

    assert [(row.label, row.source_split, row.patient_id) for row in records] == [
        (0, "train", "person12"),
        (1, "test", "person34"),
    ]


def test_rejects_image_without_patient_key(tmp_path):
    _make_kaggle_tree(tmp_path)
    (tmp_path / "data/kaggle/chest_xray/train/NORMAL/scan.jpeg").touch()

    with pytest.raises(DatasetLayoutError, match="patient ID"):
        build_kaggle_index(load_dataset_manifest(), project_root=tmp_path)
