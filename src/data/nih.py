from pathlib import Path

import pandas as pd

from src.data.labels import harmonize_nih


def build_nih_manifest(nih_dir: Path) -> pd.DataFrame:
    """Builds a manifest (path, label, + age/sex/view for the later causal
    analysis) from the NIH Data_Entry CSV, harmonized to Kaggle's binary
    pneumonia scheme. Never used for training, external validation only.
    """
    nih_dir = Path(nih_dir)
    raw = pd.read_csv(nih_dir / "Data_Entry_2017_v2020.csv")
    harmonized = harmonize_nih(raw)

    harmonized = harmonized.copy()
    harmonized["path"] = harmonized["Image Index"].apply(lambda name: str(nih_dir / "images" / name))
    harmonized = harmonized[harmonized["path"].apply(lambda p: Path(p).is_file())]

    return harmonized[["path", "label", "Patient Age", "Patient Sex", "View Position"]].reset_index(drop=True)
