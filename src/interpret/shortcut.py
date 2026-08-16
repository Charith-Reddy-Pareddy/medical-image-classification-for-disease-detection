import cv2
import numpy as np
import torch
from PIL import Image

from src.eval.metrics import compute_metrics
from src.interpret.gradcam import compute_gradcam
from src.interpret.lung_segmentation import lung_mask


@torch.no_grad()
def border_mask_accuracy_drop(model, loader, device, border_frac: float = 0.1) -> dict:
    """Masks the outer border_frac of each image (where scanner text/
    artifacts typically live) and measures the accuracy delta vs.
    unmasked. A large drop suggests reliance on border/text shortcuts
    rather than lung tissue.
    """
    model.eval()
    y_true, y_prob_orig, y_prob_masked = [], [], []
    for images, labels in loader:
        images = images.to(device)
        probs_orig = torch.sigmoid(model(images).squeeze(1)).cpu().numpy()

        masked = images.clone()
        _, _, h, w = masked.shape
        bh, bw = int(h * border_frac), int(w * border_frac)
        masked[:, :, :bh, :] = 0.0
        masked[:, :, -bh:, :] = 0.0
        masked[:, :, :, :bw] = 0.0
        masked[:, :, :, -bw:] = 0.0
        probs_masked = torch.sigmoid(model(masked).squeeze(1)).cpu().numpy()

        y_true.extend(labels.tolist())
        y_prob_orig.extend(probs_orig.tolist())
        y_prob_masked.extend(probs_masked.tolist())

    metrics_orig = compute_metrics(y_true, y_prob_orig)
    metrics_masked = compute_metrics(y_true, y_prob_masked)
    return {
        "accuracy_original": metrics_orig["accuracy"],
        "accuracy_masked": metrics_masked["accuracy"],
        "accuracy_drop": metrics_orig["accuracy"] - metrics_masked["accuracy"],
    }


def gradcam_overlap_fraction(cam: np.ndarray, mask: np.ndarray, cam_threshold: float = 0.5) -> float:
    """Fraction of the Grad-CAM heatmap's attended ("hot") region that
    falls inside the lung mask. Low overlap means the model is attending
    to something other than lung tissue.
    """
    hot = cam >= cam_threshold
    if hot.sum() == 0:
        return 0.0
    return float((hot & mask).sum()) / float(hot.sum())


def shortcut_metric_report(
    model,
    manifest_df,
    device,
    transform,
    n_samples: int | None = None,
    cam_threshold: float = 0.5,
    overlap_cutoff: float = 0.3,
    seed: int = 42,
) -> dict:
    """For correctly classified examples, computes Grad-CAM/lung-mask
    overlap and reports what fraction fall under overlap_cutoff -- the
    quantitative shortcut-learning signal from Experiment 8.
    """
    df = manifest_df
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    model.eval()
    overlaps = []
    for _, row in df.iterrows():
        image = Image.open(row["path"]).convert("RGB")
        tensor = transform(image).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor.unsqueeze(0)).squeeze(1)).item()
        pred = int(prob >= 0.5)
        if pred != row["label"]:
            continue

        cam = compute_gradcam(model, tensor, category=pred)
        mask = lung_mask(row["path"])
        mask_resized = cv2.resize(
            mask.astype(np.uint8), (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        overlaps.append(gradcam_overlap_fraction(cam, mask_resized, cam_threshold))

    if not overlaps:
        return {"n_correct": 0, "pct_under_cutoff": float("nan"), "mean_overlap": float("nan")}

    overlaps_arr = np.array(overlaps)
    return {
        "n_correct": len(overlaps),
        "pct_under_cutoff": float((overlaps_arr < overlap_cutoff).mean() * 100),
        "mean_overlap": float(overlaps_arr.mean()),
    }
