import csv
from pathlib import Path

import pytest

from chest_xray_audit.kaggle_index import DatasetLayoutError, ImageRecord
from chest_xray_audit.split_manifest import (
    read_patient_split_manifest,
    write_patient_split_manifest,
)


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


def test_reads_split_manifest_with_required_columns(tmp_path):
    path = tmp_path / "splits.csv"
    path.write_text(
        "image_path,patient_id,label,source_split,assigned_split\n"
        "train/NORMAL/person12.jpeg,person12,0,train,train\n",
        encoding="utf-8",
    )

    rows = read_patient_split_manifest(path)

    assert rows == [
        {
            "image_path": "train/NORMAL/person12.jpeg",
            "patient_id": "person12",
            "label": "0",
            "source_split": "train",
            "assigned_split": "train",
        }
    ]


def test_reader_rejects_path_traversal(tmp_path):
    path = tmp_path / "unsafe.csv"
    path.write_text(
        "image_path,patient_id,label,source_split,assigned_split\n"
        "../data/train/NORMAL/person12.jpeg,person12,0,train,train\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetLayoutError, match="invalid relative image path"):
        read_patient_split_manifest(path)


def test_reader_rejects_patient_leakage(tmp_path):
    path = tmp_path / "leakage.csv"
    path.write_text(
        "image_path,patient_id,label,source_split,assigned_split\n"
        "train/NORMAL/person12_1.jpeg,person12,0,train,train\n"
        "test/PNEUMONIA/person12_2.jpeg,person12,1,test,val\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetLayoutError, match="Patient leakage"):
        read_patient_split_manifest(path)


def test_reader_rejects_missing_image_under_root(tmp_path):
    path = tmp_path / "splits.csv"
    path.write_text(
        "image_path,patient_id,label,source_split,assigned_split\n"
        "train/NORMAL/person12.jpeg,person12,0,train,train\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetLayoutError, match="image file not found"):
        read_patient_split_manifest(path, image_root=tmp_path / "images")
