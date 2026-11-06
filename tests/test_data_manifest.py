import pytest

from chest_xray_audit.data_manifest import ManifestError, load_dataset_manifest


def test_load_dataset_manifest():
    manifest = load_dataset_manifest()

    assert manifest["schema_version"] == 1
    assert set(manifest["datasets"]) == {"kaggle", "nih", "openi"}
    assert manifest["datasets"]["nih"]["group_key"] == "Patient ID"


def test_rejects_unsupported_schema_version(tmp_path):
    path = tmp_path / "datasets.toml"
    path.write_text('schema_version = 2\ndata_root = "data"\n', encoding="utf-8")

    with pytest.raises(ManifestError, match="schema_version"):
        load_dataset_manifest(path)


def test_rejects_dataset_path_outside_repository(tmp_path):
    path = tmp_path / "datasets.toml"
    path.write_text(
        'schema_version = 1\ndata_root = "data"\n'
        '[datasets.sample]\nrole = "train"\nimage_dir = "../images"\n'
        'group_unit = "patient"\ngroup_key = "patient_id"\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="within the repository"):
        load_dataset_manifest(path)
