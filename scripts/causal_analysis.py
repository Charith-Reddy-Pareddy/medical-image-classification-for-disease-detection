import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import IMAGE_SIZE, KAGGLE_DIR, MODEL_DIR, NIH_DIR, SEED, get_device
from src.data.dataset import get_transforms
from src.data.nih import build_nih_manifest
from src.data.split import build_manifest, patient_level_split
from src.interpret.causal import build_causal_dataset, fit_shortcut_logistic_regression
from src.interpret.shortcut import per_image_overlaps
from src.models import MODEL_REGISTRY


def main(model_name: str, kaggle_n: int, nih_n: int):
    device = get_device()

    model = MODEL_REGISTRY[model_name]()
    model.load_state_dict(torch.load(MODEL_DIR / f"{model_name}.pt", map_location=device))
    model.to(device)
    model.eval()

    transform = get_transforms(IMAGE_SIZE, train=False)

    kaggle_manifest = build_manifest(KAGGLE_DIR)
    _train_df, _val_df, kaggle_test = patient_level_split(kaggle_manifest, seed=SEED)
    nih_manifest = build_nih_manifest(NIH_DIR)

    print(f"Computing pediatric (Kaggle) overlaps, sampling up to {kaggle_n}...")
    kaggle_records = per_image_overlaps(model, kaggle_test, device, transform, n_samples=kaggle_n, seed=1)
    print(f"  {len(kaggle_records)} correctly classified")

    print(f"Computing adult (NIH) overlaps, sampling up to {nih_n} (this is the slow one)...")
    nih_records = per_image_overlaps(model, nih_manifest, device, transform, n_samples=nih_n, seed=1)
    print(f"  {len(nih_records)} correctly classified")

    shortcut_records = [{"path": r["path"], "overlap": r["overlap"], "age_group": 0} for r in kaggle_records] + [
        {"path": r["path"], "overlap": r["overlap"], "age_group": 1} for r in nih_records
    ]

    if len(shortcut_records) < 20:
        print(f"\nOnly {len(shortcut_records)} correctly classified examples total -- too few for a "
              "meaningful logistic regression (need more per architecture's accuracy on each site). "
              "Increase --kaggle-n/--nih-n or use a better-trained checkpoint.")
        return

    causal_df = build_causal_dataset(shortcut_records)
    print("\n=== Group summary ===")
    print(causal_df.groupby("age_group")[["shortcut_driven", "text_marker", "log_area", "aspect_ratio"]].mean())

    print("\n=== Logistic regression: shortcut_driven ~ age_group + log_area + aspect_ratio + text_marker ===")
    try:
        result = fit_shortcut_logistic_regression(causal_df)
        print(result.summary())
    except Exception as e:
        print(f"Regression failed to fit cleanly ({e}). With this few examples and this checkpoint's "
              "heavy skew toward one class (see Day 5's domain-shift results), quasi-separation is "
              "expected -- rerun with more samples or a properly converged model for a trustworthy fit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline_cnn", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--kaggle-n", type=int, default=60)
    parser.add_argument("--nih-n", type=int, default=400)
    args = parser.parse_args()
    main(args.model, args.kaggle_n, args.nih_n)
