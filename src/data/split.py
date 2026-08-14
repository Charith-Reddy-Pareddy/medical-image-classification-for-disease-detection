import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

LABEL_NORMAL = 0
LABEL_PNEUMONIA = 1

_PERSON_RE = re.compile(r"^person(\d+)_")
_IM_RE = re.compile(r"^(NORMAL2-)?IM-(\d+)")


def parse_patient_id(filename: str) -> str:
    """Recover a patient identifier from a Kaggle chest-xray filename.

    PNEUMONIA images are named person<id>_bacteria|virus_<n>.jpeg, so the
    person id is the patient key. NORMAL images don't have that prefix but
    do repeat an IM-<id> (or NORMAL2-IM-<id>) prefix across images of the
    same patient, so that prefix is used instead.
    """
    stem = Path(filename).stem

    m = _PERSON_RE.match(stem)
    if m:
        return f"person{m.group(1)}"

    m = _IM_RE.match(stem)
    if m:
        prefix = m.group(1) or ""
        return f"{prefix}IM-{m.group(2)}"

    # Unrecognized naming convention: treat as its own patient rather than
    # silently grouping unrelated images together.
    return stem


def build_manifest(kaggle_dir: Path) -> pd.DataFrame:
    """Scan the Kaggle chest_xray folder into a flat manifest.

    All of train/test/val are combined here — the official split is
    discarded in favor of the patient-level re-split in
    `patient_level_split`, since Kaggle's split leaks patients across
    train/test.
    """
    kaggle_dir = Path(kaggle_dir)
    rows = []
    for split_dir in ("train", "test", "val"):
        for class_name, label in (("NORMAL", LABEL_NORMAL), ("PNEUMONIA", LABEL_PNEUMONIA)):
            folder = kaggle_dir / split_dir / class_name
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in (".jpeg", ".jpg", ".png"):
                    continue
                rows.append(
                    {
                        "path": str(path),
                        "label": label,
                        "patient_id": parse_patient_id(path.name),
                    }
                )
    return pd.DataFrame(rows, columns=["path", "label", "patient_id"])


def patient_level_split(df: pd.DataFrame, test_size=0.15, val_size=0.1, seed=42):
    """Split a manifest into train/val/test with no patient in more than one split."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(df, groups=df["patient_id"]))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    relative_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_idx, val_idx = next(gss_val.split(trainval_df, groups=trainval_df["patient_id"]))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def assert_no_patient_leakage(*splits: pd.DataFrame) -> None:
    seen = set()
    for split_df in splits:
        patients = set(split_df["patient_id"])
        overlap = seen & patients
        if overlap:
            raise ValueError(f"Patient leakage across splits: {sorted(overlap)[:5]}")
        seen |= patients
