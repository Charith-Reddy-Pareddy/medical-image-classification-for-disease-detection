import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import IMAGE_SIZE, KAGGLE_DIR, MODEL_DIR, SEED, get_device
from src.data.dataset import ChestXrayDataset, get_transforms
from src.data.split import build_manifest, patient_level_split
from src.interpret.failure_taxonomy import build_failure_taxonomy
from src.interpret.shortcut import border_mask_accuracy_drop, shortcut_metric_report
from src.models import MODEL_REGISTRY


def main(model_name: str, n_samples: int | None):
    device = get_device()

    model = MODEL_REGISTRY[model_name]()
    model.load_state_dict(torch.load(MODEL_DIR / f"{model_name}.pt", map_location=device))
    model.to(device)
    model.eval()

    manifest = build_manifest(KAGGLE_DIR)
    _train_df, _val_df, test_df = patient_level_split(manifest, seed=SEED)
    transform = get_transforms(IMAGE_SIZE, train=False)

    print("=== Quantitative shortcut metric (Grad-CAM vs. lung segmentation) ===")
    print(shortcut_metric_report(model, test_df, device, transform, n_samples=n_samples))

    print("\n=== Border/text mask accuracy drop ===")
    sample_df = test_df if n_samples is None else test_df.sample(n=min(n_samples, len(test_df)), random_state=SEED)
    loader = DataLoader(ChestXrayDataset(sample_df, transform=transform), batch_size=32)
    print(border_mask_accuracy_drop(model, loader, device, border_frac=0.1))

    print("\n=== False-negative failure taxonomy ===")
    taxonomy = build_failure_taxonomy(model, test_df, device, transform, n_samples=n_samples)
    print(taxonomy["category"].value_counts() if len(taxonomy) else "no false negatives in sample")


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
