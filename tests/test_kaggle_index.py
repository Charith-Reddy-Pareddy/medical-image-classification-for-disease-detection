from pathlib import Path

import pytest

from chest_xray_audit.data_manifest import load_dataset_manifest
from chest_xray_audit.kaggle_index import (
    DatasetLayoutError,
    ImageRecord,
    assign_patient_splits,
    build_kaggle_index,
    find_patient_split_overlaps,
    group_records_by_patient,
)


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


def test_reports_patients_in_multiple_source_splits():
    records = [
        ImageRecord(Path("train/person12_1.jpeg"), 0, "train", "person12"),
        ImageRecord(Path("train/person12_2.jpeg"), 1, "train", "person12"),
        ImageRecord(Path("test/person12_3.jpeg"), 0, "test", "person12"),
        ImageRecord(Path("val/person34_1.jpeg"), 1, "val", "person34"),
    ]

    assert find_patient_split_overlaps(records) == {"person12": ("test", "train")}


def test_groups_all_patient_records_across_source_splits():
    records = [
        ImageRecord(Path("train/person12_1.jpeg"), 0, "train", "person12"),
        ImageRecord(Path("test/person12_2.jpeg"), 1, "test", "person12"),
        ImageRecord(Path("train/person34_1.jpeg"), 0, "train", "person34"),
    ]

    groups = group_records_by_patient(records)

    assert list(groups) == ["person12", "person34"]
    assert groups["person12"] == (records[0], records[1])
    assert groups["person34"] == (records[2],)


def test_assigns_patient_groups_deterministically_and_by_ratio():
    records = [
        ImageRecord(Path("train/person1_a.jpeg"), 0, "train", "person1"),
        ImageRecord(Path("test/person1_b.jpeg"), 1, "test", "person1"),
        *(
            ImageRecord(Path(f"train/person{n}_a.jpeg"), n % 2, "train", f"person{n}")
            for n in range(2, 6)
        ),
    ]
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}

    assigned = assign_patient_splits(records, ratios, seed=17)

    assert assigned == assign_patient_splits(list(reversed(records)), ratios, seed=17)
    assert set(assigned) == {f"person{n}" for n in range(1, 6)}
    assert assigned["person1"] in ratios
    assert {split: list(assigned.values()).count(split) for split in ratios} == {
        "train": 3,
        "val": 1,
        "test": 1,
    }


def test_stratified_split_keeps_patient_groups_together_and_balances_labels():
    records = [
        ImageRecord(Path(f"{label}/person{label}{n}.jpeg"), label, "train", f"person{label}{n}")
        for label in (0, 1)
        for n in range(12)
    ]
    records.extend(
        [
            ImageRecord(Path("train/person-mixed-normal.jpeg"), 0, "train", "person-mixed"),
            ImageRecord(Path("test/person-mixed-pneumonia.jpeg"), 1, "test", "person-mixed"),
        ]
    )
    ratios = {"train": 0.5, "val": 0.25, "test": 0.25}

    assigned = assign_patient_splits(records, ratios, seed=21, stratify_by_label=True)

    assert assigned == assign_patient_splits(
        list(reversed(records)), ratios, seed=21, stratify_by_label=True
    )
    for label in (0, 1):
        counts = {
            split: sum(assigned[f"person{label}{n}"] == split for n in range(12))
            for split in ratios
        }
        assert counts == {"train": 6, "val": 3, "test": 3}
    assert assigned["person-mixed"] in ratios
