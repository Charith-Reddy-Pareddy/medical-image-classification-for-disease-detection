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
        (data_root / "kaggle/chest_xray" / split).mkdir(parents=True)
    (data_root / "nih/images").mkdir(parents=True)
    (data_root / "nih/Data_Entry_2017_v2020.csv").touch()
    (data_root / "openi/images").mkdir(parents=True)
    (data_root / "openi/reports/ecgen-radiology").mkdir(parents=True)

    result = main(["--data-root", str(data_root)])

    assert result == 0
    assert capsys.readouterr().out == "All dataset inputs are present.\n"
