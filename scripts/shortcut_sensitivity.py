"""Sensitivity analysis for the quantitative shortcut-learning metric
(Experiment 8): how much do the headline "74-81% of correct predictions
show low lung overlap" numbers depend on the two thresholds baked into
that metric -- cam_threshold (which Grad-CAM pixels count as "hot") and
overlap_cutoff (how little lung overlap counts as "shortcut-driven")?

Sweeps both thresholds and reports the percent shortcut-driven under
every combination as a table and a heatmap. If the conclusion holds
across the whole reasonable range, that's a stronger claim than a single
cam_threshold=0.5/overlap_cutoff=0.3 number can support on its own.

Grad-CAM/lung-segmentation is only re-run once per cam_threshold (not once
per (cam_threshold, overlap_cutoff) pair) via
`per_image_overlaps_multi_threshold`, which reuses each image's raw CAM
and lung mask across every overlap_cutoff -- see src/interpret/shortcut.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import IMAGE_SIZE, KAGGLE_DIR, MODEL_DIR, ROOT, SEED, get_device
from src.data.dataset import get_transforms
from src.data.split import build_manifest, patient_level_split
from src.interpret.shortcut import per_image_overlaps_multi_threshold
from src.models import MODEL_REGISTRY

ASSETS_DIR = ROOT / "docs" / "report_assets"

CAM_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
OVERLAP_CUTOFFS = [0.2, 0.3, 0.4, 0.5]


def compute_sensitivity_table(model_name: str, n_samples: int | None) -> pd.DataFrame:
    device = get_device()
    model = MODEL_REGISTRY[model_name]()
    model.load_state_dict(torch.load(MODEL_DIR / f"{model_name}.pt", map_location=device))
    model.to(device)
    model.eval()

    manifest = build_manifest(KAGGLE_DIR)
    _train_df, _val_df, test_df = patient_level_split(manifest, seed=SEED)
    transform = get_transforms(IMAGE_SIZE, train=False)

    print(f"Computing Grad-CAM/lung overlap at cam_thresholds={CAM_THRESHOLDS} (single pass each)...")
    records_by_cam_threshold = per_image_overlaps_multi_threshold(
        model, test_df, device, transform, CAM_THRESHOLDS, n_samples=n_samples, seed=SEED
    )

    rows = []
    for cam_threshold, records in records_by_cam_threshold.items():
        overlaps = np.array([r["overlap"] for r in records])
        row = {"cam_threshold": cam_threshold, "n_correct": len(records)}
        for cutoff in OVERLAP_CUTOFFS:
            pct = float((overlaps < cutoff).mean() * 100) if len(overlaps) else float("nan")
            row[f"overlap_cutoff={cutoff}"] = pct
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cam_threshold").reset_index(drop=True)


def plot_heatmap(table: pd.DataFrame, model_name: str, out_path: Path):
    cutoff_cols = [c for c in table.columns if c.startswith("overlap_cutoff=")]
    matrix = table[cutoff_cols].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cutoff_cols)))
    ax.set_xticklabels([c.split("=")[1] for c in cutoff_cols])
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table["cam_threshold"])
    ax.set_xlabel("overlap_cutoff")
    ax.set_ylabel("cam_threshold")
    ax.set_title(f"% correct predictions classified shortcut-driven\n({model_name})")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > 55 else "black"
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, label="% shortcut-driven")
    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(model_name: str, n_samples: int | None):
    table = compute_sensitivity_table(model_name, n_samples)
    print("\n=== Shortcut-metric sensitivity: % correct predictions under overlap_cutoff ===")
    print(table.to_string(index=False))

    cutoff_cols = [c for c in table.columns if c.startswith("overlap_cutoff=")]
    values = table[cutoff_cols].to_numpy()
    print(f"\nRange across the full sweep: {values.min():.1f}%-{values.max():.1f}%")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ASSETS_DIR / f"shortcut_sensitivity_{model_name}.csv"
    table.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    png_path = ASSETS_DIR / f"shortcut_sensitivity_{model_name}.png"
    plot_heatmap(table, model_name, png_path)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline_cnn", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Cap on the Kaggle test set used (for fast iteration). Omit for the full set.",
    )
    args = parser.parse_args()
    main(args.model, args.n_samples)
