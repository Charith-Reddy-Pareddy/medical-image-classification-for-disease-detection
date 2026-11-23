from chest_xray_audit.preflight import main


def test_cli_reports_missing_inputs(tmp_path, capsys):
    result = main(["--data-root", str(tmp_path / "not-downloaded")])

    output = capsys.readouterr().out
    assert result == 1
    assert "Missing dataset inputs:" in output
    assert str(tmp_path / "not-downloaded/kaggle/chest_xray/train") in output


def test_cli_succeeds_when_inputs_exist(tmp_path, capsys):
    data_root = tmp_path / "datasets"
    for split in ("train", "val", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            (data_root / "kaggle/chest_xray" / split / label).mkdir(parents=True)
    (data_root / "nih/images").mkdir(parents=True)
    (data_root / "nih/Data_Entry_2017_v2020.csv").touch()
    (data_root / "openi/images").mkdir(parents=True)
    (data_root / "openi/reports/ecgen-radiology").mkdir(parents=True)
    (data_root / "kaggle/chest_xray/train/NORMAL/person12_1.jpeg").touch()
    (data_root / "kaggle/chest_xray/test/PNEUMONIA/person12_bacteria_2.jpeg").touch()

    result = main(["--data-root", str(data_root)])

    assert result == 0
    assert capsys.readouterr().out == (
        "All dataset inputs are present.\n"
        "Indexed 2 Kaggle images.\n"
        "Patients in multiple source splits: 1\n"
        "Patient split settings: seed=42, stratify_by_label=True, "
        "ratios=train=0.7, val=0.15, test=0.15\n"
        "Assigned patient-level splits:\n"
        "  train: 1 patients, 2 images (NORMAL=1, PNEUMONIA=1)\n"
        "  val: 0 patients, 0 images (NORMAL=0, PNEUMONIA=0)\n"
        "  test: 0 patients, 0 images (NORMAL=0, PNEUMONIA=0)\n"
    )


def test_cli_uses_manifest_patient_split_settings(tmp_path, capsys, monkeypatch):
    from chest_xray_audit import preflight
    from chest_xray_audit.data_manifest import load_dataset_manifest

    data_root = tmp_path / "datasets"
    for split in ("train", "val", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            (data_root / "kaggle/chest_xray" / split / label).mkdir(parents=True)
    (data_root / "nih/images").mkdir(parents=True)
    (data_root / "nih/Data_Entry_2017_v2020.csv").touch()
    (data_root / "openi/images").mkdir(parents=True)
    (data_root / "openi/reports/ecgen-radiology").mkdir(parents=True)
    (data_root / "kaggle/chest_xray/train/NORMAL/person12_1.jpeg").touch()
    (data_root / "kaggle/chest_xray/test/PNEUMONIA/person34_bacteria_2.jpeg").touch()

    manifest = load_dataset_manifest()
    settings = {
        "ratios": {"train": 0.0, "val": 1.0, "test": 0.0},
        "seed": 17,
        "stratify_by_label": False,
    }
    manifest["patient_split"] = settings
    monkeypatch.setattr(preflight, "load_dataset_manifest", lambda: manifest)

    passed = {}
    assign = preflight.assign_records_to_splits

    def capture(records, ratios=None, *, seed=42, stratify_by_label=False):
        passed.update(ratios=ratios, seed=seed, stratify_by_label=stratify_by_label)
        return assign(
            records,
            ratios=ratios,
            seed=seed,
            stratify_by_label=stratify_by_label,
        )

    monkeypatch.setattr(preflight, "assign_records_to_splits", capture)

    result = preflight.main(["--data-root", str(data_root)])

    assert result == 0
    assert passed == settings
    assert "  val: 2 patients, 2 images (NORMAL=1, PNEUMONIA=1)" in capsys.readouterr().out
