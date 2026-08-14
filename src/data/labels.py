"""Map NIH ChestX-ray14 and CheXpert's native labels onto Kaggle's binary
NORMAL/PNEUMONIA scheme, per the harmonization protocol in docs/roadmap.md.
Kaggle needs no mapping -- it's already binary.
"""
import pandas as pd


def harmonize_nih(df: pd.DataFrame) -> pd.DataFrame:
    """NIH ChestX-ray14 is multi-label. label=1 if "Pneumonia" is among the
    Finding Labels (co-occurring findings allowed); label=0 if Finding
    Labels is exactly "No Finding". Anything else (another finding, no
    pneumonia) is an ambiguous negative and is excluded.
    """

    def to_label(finding_labels: str):
        if finding_labels.strip() == "No Finding":
            return 0
        if "Pneumonia" in finding_labels.split("|"):
            return 1
        return None

    out = df.copy()
    out["label"] = out["Finding Labels"].apply(to_label)
    return out.dropna(subset=["label"]).astype({"label": int})


def harmonize_chexpert(df: pd.DataFrame, uncertain_policy: str = "ignore") -> pd.DataFrame:
    """CheXpert's Pneumonia column is 1 / 0 / -1 (uncertain) / NaN (unmentioned).

    label=1 if Pneumonia == 1
    label=0 if Pneumonia == 0, or "No Finding" == 1
    Pneumonia == -1 (uncertain) is excluded under "ignore" (U-Ignore, the
    primary policy) or mapped to 1 under "ones" (U-Ones sensitivity check).
    Evaluation is restricted to frontal views to match Kaggle/NIH.
    """
    if uncertain_policy not in ("ignore", "ones"):
        raise ValueError(f"unknown uncertain_policy: {uncertain_policy!r}")

    out = df.copy()
    if "Frontal/Lateral" in out.columns:
        out = out[out["Frontal/Lateral"] == "Frontal"]

    pneumonia = out["Pneumonia"]
    no_finding = out["No Finding"].fillna(0) if "No Finding" in out.columns else 0

    label = pd.Series(pd.NA, index=out.index, dtype="Int64")
    label[(pneumonia == 0) | (no_finding == 1)] = 0
    label[pneumonia == 1] = 1
    if uncertain_policy == "ones":
        label[pneumonia == -1] = 1

    out = out.assign(label=label)
    return out.dropna(subset=["label"]).astype({"label": int})
