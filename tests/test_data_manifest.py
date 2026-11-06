import pytest

from chest_xray_audit.data_manifest import (
    ManifestError,
    load_dataset_manifest,
    resolve_dataset_paths,
)


def test_load_dataset_manifest():
    manifest = load_dataset_manifest()

    assert manifest["schema_version"] == 1
    assert set(manifest["datasets"]) == {"kaggle", "nih", "openi"}
    assert manifest["datasets"]["nih"]["group_key"] == "Patient ID"


def test_resolves_paths_under_project_data_root(tmp_path):
    manifest = load_dataset_manifest()

    paths = resolve_dataset_paths(manifest, project_root=tmp_path)

    assert paths["nih"]["image_dir"] == (tmp_path / "data/nih/images").resolve()
    assert (
        paths["nih"]["metadata_file"] == (tmp_path / "data/nih/Data_Entry_2017_v2020.csv").resolve()
    )


def test_resolves_paths_under_custom_data_root(tmp_path):
    manifest = load_dataset_manifest()
    data_root = tmp_path / "private-datasets"

    paths = resolve_dataset_paths(manifest, data_root=data_root)

    assert paths["openi"]["image_dir"] == (data_root / "openi/images").resolve()


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
