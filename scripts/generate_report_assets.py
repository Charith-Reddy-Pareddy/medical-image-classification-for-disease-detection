"""Generates the report's negative-case gallery (misclassified images with
Grad-CAM overlays) and the hero domain-shift visualization. Run after
scripts/generate_results_table.py so docs/results.md is current.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import IMAGE_SIZE, KAGGLE_DIR, MODEL_DIR, ROOT, SEED, get_device
from src.data.dataset import get_transforms
from src.data.split import build_manifest, patient_level_split
from src.interpret.gradcam import compute_gradcam
from src.models import MODEL_REGISTRY

ASSETS_DIR = ROOT / "docs" / "report_assets"
RESULTS_PATH = ROOT / "docs" / "results.md"


def generate_negative_case_gallery(model_name: str, n_cases: int = 5, seed: int = SEED):
    device = get_device()
    model = MODEL_REGISTRY[model_name]()
    model.load_state_dict(torch.load(MODEL_DIR / f"{model_name}.pt", map_location=device))
    model.to(device)
    model.eval()

    manifest = build_manifest(KAGGLE_DIR)
    _train_df, _val_df, test_df = patient_level_split(manifest, seed=SEED)
    transform = get_transforms(IMAGE_SIZE, train=False)

    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    found = []
    for _, row in test_df.iterrows():
        image = Image.open(row["path"]).convert("RGB")
        tensor = transform(image).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor.unsqueeze(0)).squeeze(1)).item()
        pred = int(prob >= 0.5)
        if pred == row["label"]:
            continue  # only misclassified cases

        cam = compute_gradcam(model, tensor, category=pred)
        display_img = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
        overlay = show_cam_on_image(display_img, cam, use_rgb=True)

        true_label = "PNEUMONIA" if row["label"] == 1 else "NORMAL"
        pred_label = "PNEUMONIA" if pred == 1 else "NORMAL"
        fname = f"{model_name}_misclassified_{len(found)+1}_true-{true_label}_pred-{pred_label}_p{prob:.2f}.png"

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(display_img)
        axes[0].set_title(f"Input (true: {true_label})", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM (pred: {pred_label}, p={prob:.2f})", fontsize=9)
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(ASSETS_DIR / fname, dpi=120)
        plt.close(fig)

        found.append({"path": row["path"], "true": true_label, "pred": pred_label, "prob": prob, "asset": fname})
        if len(found) >= n_cases:
            break

    return found


def parse_results_table(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            model, dataset, _n, acc, _prec, _rec, _f1, auc = m.groups()
            rows.append({"model": model, "dataset": dataset, "auc": float(auc), "accuracy": float(acc)})
    return rows


def generate_hero_chart():
    rows = parse_results_table(RESULTS_PATH)
    models = sorted({r["model"] for r in rows})
    datasets = ["kaggle_test", "nih", "openi"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(datasets))
    width = 0.25
    for i, model in enumerate(models):
        aucs = [next(r["auc"] for r in rows if r["model"] == model and r["dataset"] == d) for d in datasets]
        ax.bar(x + i * width, aucs, width, label=model)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (AUC=0.5)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Kaggle\n(in-domain)", "NIH\n(external)", "OpenI\n(external)"])
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Domain shift: AUC-ROC by architecture and evaluation site")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "hero_domain_shift_auc.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating hero chart...")
    generate_hero_chart()
    print("Generating negative-case gallery for baseline_cnn...")
    cases = generate_negative_case_gallery("baseline_cnn", n_cases=5)
    for c in cases:
        print(c)
    print(f"\nAssets written to {ASSETS_DIR}")
