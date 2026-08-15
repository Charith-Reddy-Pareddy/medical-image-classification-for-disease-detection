import pandas as pd
from PIL import Image

from src.data.nih import build_nih_manifest


def _make_nih_dir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    rows = [
        {"Image Index": "a.png", "Finding Labels": "Pneumonia", "Patient Age": 40, "Patient Sex": "M", "View Position": "PA"},
        {"Image Index": "b.png", "Finding Labels": "No Finding", "Patient Age": 25, "Patient Sex": "F", "View Position": "AP"},
        {"Image Index": "c.png", "Finding Labels": "Infiltration", "Patient Age": 60, "Patient Sex": "M", "View Position": "PA"},
        # d.png has a label but the file is missing on disk -- should be dropped
        {"Image Index": "d.png", "Finding Labels": "Pneumonia", "Patient Age": 50, "Patient Sex": "F", "View Position": "PA"},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "Data_Entry_2017_v2020.csv", index=False)

    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (16, 16)).save(images_dir / name)

    return tmp_path


def test_build_nih_manifest_harmonizes_and_resolves_paths(tmp_path):
    nih_dir = _make_nih_dir(tmp_path)
    manifest = build_nih_manifest(nih_dir)

    # c.png (ambiguous "Infiltration" finding) excluded by harmonize_nih,
    # d.png excluded because the image file doesn't actually exist
    assert len(manifest) == 2
    assert set(manifest["label"]) == {0, 1}
    assert all(manifest["path"].apply(lambda p: p.endswith(".png")))
