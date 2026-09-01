import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.interpret.gradcam import compute_gradcam
from src.interpret.lung_segmentation import lung_mask
from src.interpret.shortcut import gradcam_overlap_fraction

CATEGORIES = ("image_quality_issue", "shortcut_feature_driven", "borderline_ground_truth", "subtle_opacity")


def categorize_false_negative(
    prob: float,
    cam_lung_overlap: float,
    mean_intensity: float,
    borderline_band: tuple = (0.35, 0.5),
    shortcut_overlap_cutoff: float = 0.3,
    exposure_low: float = 30,
    exposure_high: float = 225,
) -> str:
    """Heuristic first-pass bucket for a false negative. Checked in order:
    exposure problems first (most objective), then shortcut reliance, then
    a near-threshold call, else treated as a genuinely subtle finding.
    Meant to triage candidates for the report's failure taxonomy
    (Experiment 6) -- hand-verify before picking example images.
    """
    if mean_intensity < exposure_low or mean_intensity > exposure_high:
        return "image_quality_issue"
    if cam_lung_overlap < shortcut_overlap_cutoff:
        return "shortcut_feature_driven"
    if borderline_band[0] <= prob < borderline_band[1]:
        return "borderline_ground_truth"
    return "subtle_opacity"


def build_failure_taxonomy(
    model, manifest_df, device, transform, n_samples: int | None = None, seed: int = 42
) -> pd.DataFrame:
    """Finds false negatives, computes Grad-CAM/lung overlap and exposure
    stats for each, and heuristically buckets them.
    """
    df = manifest_df
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    model.eval()
    rows = []
    for _, row in df.iterrows():
        if row["label"] != 1:
            continue

        image = Image.open(row["path"]).convert("RGB")
        tensor = transform(image).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor.unsqueeze(0)).squeeze(1)).item()
        if prob >= 0.5:
            continue  # correctly caught, not a false negative

        cam = compute_gradcam(model, tensor, category=0)
        mask = lung_mask(row["path"])
        mask_resized = cv2.resize(
            mask.astype(np.uint8), (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        overlap = gradcam_overlap_fraction(cam, mask_resized)
        mean_intensity = float(np.array(image.convert("L"), dtype=np.float32).mean())

        rows.append(
            {
                "path": row["path"],
                "prob": prob,
                "cam_lung_overlap": overlap,
                "mean_intensity": mean_intensity,
                "category": categorize_false_negative(prob, overlap, mean_intensity),
            }
        )

    return pd.DataFrame(rows, columns=["path", "prob", "cam_lung_overlap", "mean_intensity", "category"])
