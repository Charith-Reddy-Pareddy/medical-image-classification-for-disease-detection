import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.data.labels import harmonize_openi


def _parse_reports(reports_dir: Path) -> pd.DataFrame:
    rows = []
    for xml_path in sorted(reports_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        majors = sorted({node.text.lower() for node in root.findall(".//MeSH/major") if node.text})
        automatics = sorted({node.text.lower() for node in root.findall(".//MeSH/automatic") if node.text})
        labels_major = "|".join(majors)
        labels_automatic = "|".join(automatics)
        for image in root.findall(".//parentImage"):
            rows.append(
                {
                    "imageid": image.attrib["id"],
                    "labels_major": labels_major,
                    "labels_automatic": labels_automatic,
                }
            )
    return pd.DataFrame(rows, columns=["imageid", "labels_major", "labels_automatic"])


def build_openi_manifest(openi_dir: Path) -> pd.DataFrame:
    """Builds a manifest (path, label) from the Indiana University/OpenI
    XML radiology reports, harmonized to Kaggle's binary pneumonia scheme.
    Never used for training, external validation only.
    """
    openi_dir = Path(openi_dir)
    raw = _parse_reports(openi_dir / "reports" / "ecgen-radiology")
    harmonized = harmonize_openi(raw)

    harmonized = harmonized.copy()
    harmonized["path"] = harmonized["imageid"].apply(lambda iid: str(openi_dir / "images" / f"{iid}.png"))
    harmonized = harmonized[harmonized["path"].apply(lambda p: Path(p).is_file())]

    return harmonized[["path", "label"]].reset_index(drop=True)
