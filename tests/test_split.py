import pandas as pd
import pytest

from src.data.split import (
    assert_no_patient_leakage,
    build_manifest,
    parse_patient_id,
    patient_level_split,
)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("person1003_bacteria_2934.jpeg", "person1003"),
        ("person1003_virus_1685.jpeg", "person1003"),
        ("IM-0629-0001.jpeg", "IM-0629"),
        ("NORMAL2-IM-1345-0001.jpeg", "NORMAL2-IM-1345"),
    ],
)
def test_parse_patient_id(filename, expected):
    assert parse_patient_id(filename) == expected


def test_same_patient_different_scans_share_id():
    a = parse_patient_id("person1003_bacteria_2934.jpeg")
    b = parse_patient_id("person1003_virus_1685.jpeg")
    assert a == b


def test_parse_patient_id_five_digit_id():
    # pins a longer id explicitly -- a regex narrowed to match only a
    # single digit would silently fall through to the "unrecognized
    # filename" fallback here instead of raising, so this needs an exact
    # value check, not just "doesn't crash"
    assert parse_patient_id("person12345_bacteria_1.jpeg") == "person12345"


def _make_kaggle_dir(tmp_path):
    for split in ("train", "test", "val"):
        for cls in ("NORMAL", "PNEUMONIA"):
            (tmp_path / split / cls).mkdir(parents=True)

    # 3 pneumonia patients, 2 images each -> 6 images
    for patient in (2001, 2002, 2003):
        for i, kind in enumerate(("bacteria", "virus")):
            (tmp_path / "train" / "PNEUMONIA" / f"person{patient}_{kind}_{i}.jpeg").touch()

    # 6 normal patients, 1 image each
    for i in range(6):
        (tmp_path / "train" / "NORMAL" / f"IM-{1000 + i}-0001.jpeg").touch()

    return tmp_path


def test_build_manifest_combines_all_splits(tmp_path):
    kaggle_dir = _make_kaggle_dir(tmp_path)
    manifest = build_manifest(kaggle_dir)
    assert len(manifest) == 12
    assert set(manifest["label"].unique()) == {0, 1}


def test_patient_level_split_has_no_leakage(tmp_path):
    kaggle_dir = _make_kaggle_dir(tmp_path)
    manifest = build_manifest(kaggle_dir)

    train_df, val_df, test_df = patient_level_split(manifest, test_size=0.34, val_size=0.2, seed=0)

    assert len(train_df) + len(val_df) + len(test_df) == len(manifest)
    assert_no_patient_leakage(train_df, val_df, test_df)


def test_assert_no_patient_leakage_raises_on_overlap():
    a = pd.DataFrame({"patient_id": ["p1", "p2"]})
    b = pd.DataFrame({"patient_id": ["p2", "p3"]})
    with pytest.raises(ValueError):
        assert_no_patient_leakage(a, b)
