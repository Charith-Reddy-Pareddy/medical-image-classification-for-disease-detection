import csv
from pathlib import Path

import pytest

from chest_xray_audit.kaggle_index import DatasetLayoutError, ImageRecord
from chest_xray_audit.split_manifest import write_patient_split_manifest


def test_writes_patient_split_csv_with_relative_image_paths(tmp_path):
    records = [
        ImageRecord(tmp_path / "data/train/NORMAL/person12_1.jpeg", 0, "train", "person12", "train"),
        ImageRecord(
            tmp_path / "data/test/PNEUMONIA/person12_bacteria_2.jpeg",
            1,
            "test",
            "person12",
            "train",
        ),
        ImageRecord(tmp_path / "data/val/NORMAL/person34_1.jpeg", 0, "val", "person34", "val"),
    ]

    out = write_patient_split_manifest(records, tmp_path / "reports/splits.csv")

    with out.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert {row["image_path"] for row in rows} == {
        "train/NORMAL/person12_1.jpeg",
        "test/PNEUMONIA/person12_bacteria_2.jpeg",
        "val/NORMAL/person34_1.jpeg",
    }
    assert {row["assigned_split"] for row in rows if row["patient_id"] == "person12"} == {
        "train"
    }
    assert all(not Path(row["image_path"]).is_absolute() for row in rows)


def test_invalid_record_does_not_replace_existing_manifest(tmp_path):
    output_path = tmp_path / "splits.csv"
    output_path.write_text("previous manifest", encoding="utf-8")
    records = [
        ImageRecord(tmp_path / "train/person12.jpeg", 0, "train", "person12", "train"),
        ImageRecord(tmp_path / "test/person34.jpeg", 2, "test", "person34", "test"),
    ]

    with pytest.raises(DatasetLayoutError, match="Unsupported Kaggle label"):
        write_patient_split_manifest(records, output_path)

    assert output_path.read_text(encoding="utf-8") == "previous manifest"
    assert not list(tmp_path.glob(".splits.csv.*.tmp"))
