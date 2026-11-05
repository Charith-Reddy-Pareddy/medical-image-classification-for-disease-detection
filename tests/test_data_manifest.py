from chest_xray_audit.data_manifest import load_dataset_manifest


def test_load_dataset_manifest():
    manifest = load_dataset_manifest()

    assert manifest["schema_version"] == 1
    assert set(manifest["datasets"]) == {"kaggle", "nih", "openi"}
    assert manifest["datasets"]["nih"]["group_key"] == "Patient ID"
