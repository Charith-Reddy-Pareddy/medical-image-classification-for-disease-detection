"""Command-line check for locally available dataset inputs."""

import argparse
from pathlib import Path

from chest_xray_audit.data_manifest import (
    ManifestError,
    find_missing_dataset_inputs,
    load_dataset_manifest,
)
from chest_xray_audit.kaggle_index import (
    DatasetLayoutError,
    assign_records_to_splits,
    build_kaggle_index,
    find_patient_split_overlaps,
    summarize_assigned_splits,
)
from chest_xray_audit.split_manifest import write_patient_split_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local dataset paths from the manifest")
    parser.add_argument("--data-root", type=Path, help="Override the manifest's data directory")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Write patient-level assignments to a CSV file",
    )
    args = parser.parse_args(argv)

    manifest = load_dataset_manifest()
    missing = find_missing_dataset_inputs(manifest, data_root=args.data_root)
    if missing:
        print("Missing dataset inputs:")
        for path in missing:
            print(f"  {path}")
        return 1

    print("All dataset inputs are present.")
    try:
        records = build_kaggle_index(manifest, data_root=args.data_root)
        split_config = manifest.get("patient_split", {})
        ratios = split_config.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
        seed = split_config.get("seed", 42)
        stratify = split_config.get("stratify_by_label", True)
        assigned = assign_records_to_splits(
            records,
            ratios=ratios,
            seed=seed,
            stratify_by_label=stratify,
        )
        summary = summarize_assigned_splits(assigned)
    except (DatasetLayoutError, FileNotFoundError, ManifestError) as exc:
        print(f"Could not index Kaggle images: {exc}")
        return 1

    overlaps = find_patient_split_overlaps(records)
    print(f"Indexed {len(records)} Kaggle images.")
    if overlaps:
        print(f"Patients in multiple source splits: {len(overlaps)}")

    ratio_text = ", ".join(f"{split}={ratios[split]:g}" for split in ("train", "val", "test"))
    print(
        f"Patient split settings: seed={seed}, stratify_by_label={stratify}, "
        f"ratios={ratio_text}"
    )
    print("Assigned patient-level splits:")
    for split in ("train", "val", "test"):
        counts = summary.get(split, {})
        labels = counts.get("labels", {})
        print(
            f"  {split}: {counts.get('patients', 0)} patients, "
            f"{counts.get('images', 0)} images "
            f"(NORMAL={labels.get(0, 0)}, PNEUMONIA={labels.get(1, 0)})"
        )

    if args.split_manifest is not None:
        try:
            output_path = write_patient_split_manifest(assigned, args.split_manifest)
        except (DatasetLayoutError, OSError) as exc:
            print(f"Could not write patient split manifest: {exc}")
            return 1
        print(f"Wrote patient split manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
