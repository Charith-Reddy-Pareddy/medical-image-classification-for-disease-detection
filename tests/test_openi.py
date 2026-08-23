import pandas as pd
from PIL import Image

from src.data.labels import harmonize_openi
from src.data.openi import build_openi_manifest


def test_harmonize_openi_pneumonia_and_normal():
    df = pd.DataFrame(
        {
            "labels_major": ["normal", "opacity/lung", "cardiomegaly", "normal"],
            "labels_automatic": ["", "peribronchial pneumonia|pneumonia", "cardiomegaly", "pneumonia"],
        }
    )
    out = harmonize_openi(df)
    # row 2 ("cardiomegaly" major, no pneumonia mention) is an ambiguous
    # negative, excluded; row 3 has pneumonia in automatic tags despite a
    # "normal" major tag -- pneumonia wins
    assert len(out) == 3
    assert out["label"].tolist() == [0, 1, 1]


def _make_openi_dir(tmp_path):
    reports_dir = tmp_path / "reports" / "ecgen-radiology"
    reports_dir.mkdir(parents=True)
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    def write_report(uid, mesh_xml, image_ids):
        parent_images = "".join(f'<parentImage id="{iid}"><url>{iid}.png</url></parentImage>' for iid in image_ids)
        (reports_dir / f"{uid}.xml").write_text(
            f"<eCitation><uId id='CXR{uid}'/><MeSH>{mesh_xml}</MeSH>{parent_images}</eCitation>"
        )

    write_report(1, "<major>normal</major>", ["CXR1_IM-0001-1001"])
    write_report(2, "<automatic>pneumonia</automatic>", ["CXR2_IM-0002-1001", "CXR2_IM-0002-2001"])
    # ambiguous: some other finding, no pneumonia -- excluded
    write_report(3, "<major>cardiomegaly</major>", ["CXR3_IM-0003-1001"])
    # pneumonia mentioned but the image file is missing on disk -- excluded
    write_report(4, "<automatic>pneumonia</automatic>", ["CXR4_IM-0004-1001"])

    for name in ("CXR1_IM-0001-1001", "CXR2_IM-0002-1001", "CXR2_IM-0002-2001", "CXR3_IM-0003-1001"):
        Image.new("RGB", (16, 16)).save(images_dir / f"{name}.png")

    return tmp_path


def test_build_openi_manifest_harmonizes_and_resolves_paths(tmp_path):
    openi_dir = _make_openi_dir(tmp_path)
    manifest = build_openi_manifest(openi_dir)

    assert len(manifest) == 3  # 1 normal + 2 pneumonia images from report 2
    assert manifest["label"].tolist() == [0, 1, 1]
    assert all(manifest["path"].apply(lambda p: p.endswith(".png")))
