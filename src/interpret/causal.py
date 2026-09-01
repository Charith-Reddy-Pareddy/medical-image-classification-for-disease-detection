"""Age-artifact causal analysis (Experiment 9): does shortcut reliance
(low Grad-CAM/lung overlap, from Experiment 8) track pediatric-vs-adult
imaging differences specifically, or is it better explained by a
measurable proxy -- image resolution and detected scanner text markers?
Tested via logistic regression rather than an eyeballed correlation.

Text-marker detection is a coarse heuristic, not OCR: tesseract wasn't
available in this environment, and the proposal explicitly allows a
"simple ... template match" as the alternative. This flags small, very
bright, tightly-packed blobs in the image corners (where scanner
timestamps/labels typically sit) -- it will miss dim or unusually placed
markers and can false-positive on genuinely overexposed corners. Treat it
as a rough proxy, not ground truth; swapping in real OCR later only
requires replacing `detect_text_marker`.
"""

import cv2
import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image

CORNER_FRAC = 0.12
BRIGHTNESS_THRESHOLD = 245
MIN_BLOB_FRAC = 0.00002
MAX_BLOB_FRAC = 0.002
MIN_PLAUSIBLE_BLOBS = 2


def image_resolution_features(path: str) -> dict:
    with Image.open(path) as img:
        width, height = img.size
    return {
        "width": width,
        "height": height,
        "log_area": float(np.log(width * height)),
        "aspect_ratio": width / height,
    }


def detect_text_marker(path: str) -> bool:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    ch, cw = int(h * CORNER_FRAC), int(w * CORNER_FRAC)

    corners = np.zeros_like(img, dtype=bool)
    corners[:ch, :cw] = True
    corners[:ch, -cw:] = True
    corners[-ch:, :cw] = True
    corners[-ch:, -cw:] = True

    hits = (img > BRIGHTNESS_THRESHOLD) & corners
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(hits.astype(np.uint8))
    total_px = h * w
    blob_areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else []
    plausible = [a for a in blob_areas if MIN_BLOB_FRAC * total_px < a < MAX_BLOB_FRAC * total_px]
    return len(plausible) >= MIN_PLAUSIBLE_BLOBS


def build_causal_dataset(shortcut_records: list) -> pd.DataFrame:
    """shortcut_records: list of dicts with 'path', 'overlap', 'age_group'
    (0=pediatric/Kaggle, 1=adult/NIH), as produced by shortcut_metric_report
    for individual images. Adds resolution and text-marker features.
    """
    rows = []
    for rec in shortcut_records:
        features = image_resolution_features(rec["path"])
        features["text_marker"] = int(detect_text_marker(rec["path"]))
        features["age_group"] = rec["age_group"]
        features["shortcut_driven"] = int(rec["overlap"] < 0.3)
        rows.append(features)
    return pd.DataFrame(rows)


def fit_shortcut_logistic_regression(causal_df: pd.DataFrame):
    """Logistic regression of shortcut_driven ~ age_group + log_area +
    aspect_ratio + text_marker. Returns the fitted statsmodels result --
    if age_group's coefficient stops being significant once the
    resolution/text-marker proxies are included, that's evidence the
    proxies (not an unexplained "age" label) are the real mechanism.
    """
    X = causal_df[["age_group", "log_area", "aspect_ratio", "text_marker"]].astype(float)
    X = sm.add_constant(X)
    y = causal_df["shortcut_driven"].astype(float)
    return sm.Logit(y, X).fit(disp=0)
