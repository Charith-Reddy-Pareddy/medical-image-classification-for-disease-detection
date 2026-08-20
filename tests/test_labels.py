import pandas as pd

from src.data.labels import harmonize_chexpert, harmonize_nih


def test_harmonize_nih_pneumonia_positive():
    df = pd.DataFrame({"Finding Labels": ["Pneumonia", "Infiltration|Pneumonia", "No Finding", "Infiltration"]})
    out = harmonize_nih(df)
    # the lone "Infiltration" row (non-pneumonia finding) is an ambiguous negative, excluded
    assert len(out) == 3
    assert out["label"].tolist() == [1, 1, 0]


def test_harmonize_chexpert_basic_labels():
    df = pd.DataFrame(
        {
            "Frontal/Lateral": ["Frontal", "Frontal", "Frontal", "Lateral"],
            "Pneumonia": [1, 0, float("nan"), 1],
            "No Finding": [0, 0, 1, 0],
        }
    )
    out = harmonize_chexpert(df)
    # lateral view is dropped, leaving 3 frontal rows
    assert len(out) == 3
    assert out["label"].tolist() == [1, 0, 0]


def test_harmonize_chexpert_uncertain_policies():
    df = pd.DataFrame(
        {
            "Frontal/Lateral": ["Frontal"],
            "Pneumonia": [-1],
            "No Finding": [0],
        }
    )
    assert len(harmonize_chexpert(df, uncertain_policy="ignore")) == 0

    ones = harmonize_chexpert(df, uncertain_policy="ones")
    assert ones["label"].tolist() == [1]


def test_harmonize_chexpert_uncertain_ones_only_affects_uncertain_rows():
    # an already-certain row alongside an uncertain one -- a regression
    # that checks pneumonia == 1 instead of == -1 for the "ones" policy
    # would happen to leave the certain row looking right, so it can't
    # be caught by a single-row case alone
    df = pd.DataFrame(
        {
            "Frontal/Lateral": ["Frontal", "Frontal"],
            "Pneumonia": [1, -1],
            "No Finding": [0, 0],
        }
    )
    out = harmonize_chexpert(df, uncertain_policy="ones")
    assert len(out) == 2
    assert out["label"].tolist() == [1, 1]
