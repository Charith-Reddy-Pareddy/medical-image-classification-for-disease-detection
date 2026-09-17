import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import (
    IMAGE_SIZE,
    KAGGLE_DIR,
    MODEL_DIR,
    NIH_DIR,
    OPENI_DIR,
    SEED,
    get_device,
)
from src.data.dataset import ChestXrayDataset, get_transforms
from src.data.nih import build_nih_manifest
from src.data.openi import build_openi_manifest
from src.data.split import build_manifest as build_kaggle_manifest
from src.data.split import parse_patient_id, patient_level_split
from src.eval.domain_shift import evaluate_across_datasets
from src.eval.stats import bootstrap_ci, bootstrap_ci_clustered, sensitivity_at_threshold
from src.models import MODEL_REGISTRY


def _bootstrap(result: dict, dataset_groups, fn, n_boot: int):
    """Patient/study-level (clustered) CI when a grouping key is available,
    example-level otherwise. Returns (point_estimate, lower, upper, kind).
    """
    if dataset_groups is not None:
        point, lo, hi = bootstrap_ci_clustered(
            result["y_true"], result["y_prob"], dataset_groups, fn, n_boot=n_boot
        )
        return point, lo, hi, "patient-level"
    point, lo, hi = bootstrap_ci(result["y_true"], result["y_prob"], fn, n_boot=n_boot)
    return point, lo, hi, "example-level"


def build_eval_loaders(batch_size: int, num_workers: int, nih_sample_size: int | None, seed: int):
    """Returns (loaders, groups): groups[name] is a patient/study-ID array
    aligned with each loader's (shuffle=False) iteration order, or None
    where no grouping key is available, for the clustered bootstrap CI.
    """
    kaggle_manifest = build_kaggle_manifest(KAGGLE_DIR)
    _train_df, _val_df, kaggle_test_df = patient_level_split(kaggle_manifest, seed=SEED)
    kaggle_test_ds = ChestXrayDataset(kaggle_test_df, transform=get_transforms(IMAGE_SIZE, train=False))
    kaggle_groups = kaggle_test_df["path"].apply(lambda p: parse_patient_id(Path(p).name)).tolist()

    nih_manifest = build_nih_manifest(NIH_DIR)
    if nih_sample_size is not None and nih_sample_size < len(nih_manifest):
        nih_manifest = nih_manifest.sample(n=nih_sample_size, random_state=seed).reset_index(drop=True)
    nih_ds = ChestXrayDataset(nih_manifest, transform=get_transforms(IMAGE_SIZE, train=False))
    nih_groups = nih_manifest["Patient ID"].tolist() if "Patient ID" in nih_manifest.columns else None

    openi_manifest = build_openi_manifest(OPENI_DIR)
    openi_ds = ChestXrayDataset(openi_manifest, transform=get_transforms(IMAGE_SIZE, train=False))
    openi_groups = openi_manifest["study_id"].tolist() if "study_id" in openi_manifest.columns else None

    loaders = {
        "kaggle_test": DataLoader(
            kaggle_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "nih": DataLoader(nih_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "openi": DataLoader(openi_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    groups = {"kaggle_test": kaggle_groups, "nih": nih_groups, "openi": openi_groups}
    return loaders, groups


def main(model_name: str, batch_size: int, num_workers: int, nih_sample_size: int | None):
    device = get_device()

    model = MODEL_REGISTRY[model_name]()
    ckpt_path = MODEL_DIR / f"{model_name}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    loaders, groups = build_eval_loaders(batch_size, num_workers, nih_sample_size, SEED)
    results = evaluate_across_datasets(model, loaders, device)

    for name, result in results.items():
        print(f"\n=== {name} (n={len(result['y_true'])}) ===")
        print("metrics:", result["metrics"])

        dataset_groups = groups.get(name)

        auc_point, auc_lo, auc_hi, ci_kind = _bootstrap(result, dataset_groups, roc_auc_score, 1000)
        print(f"AUC-ROC: {auc_point:.3f} ({ci_kind} 95% CI {auc_lo:.3f}-{auc_hi:.3f})")

        sens_fn = sensitivity_at_threshold(0.5)
        sens_point, sens_lo, sens_hi, ci_kind = _bootstrap(result, dataset_groups, sens_fn, 1000)
        print(f"Sensitivity: {sens_point:.3f} ({ci_kind} 95% CI {sens_lo:.3f}-{sens_hi:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline_cnn", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--nih-sample-size",
        type=int,
        default=None,
        help="Evaluate on a random sample of NIH instead of all 61k+ examples (for fast iteration). Omit for the full set.",
    )
    args = parser.parse_args()
    main(args.model, args.batch_size, args.num_workers, args.nih_sample_size)
