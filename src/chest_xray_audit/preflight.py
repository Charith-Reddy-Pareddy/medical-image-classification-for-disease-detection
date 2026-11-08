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
    build_kaggle_index,
    find_patient_split_overlaps,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local dataset paths from the manifest")
    parser.add_argument("--data-root", type=Path, help="Override the manifest's data directory")
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
    except (DatasetLayoutError, FileNotFoundError, ManifestError) as exc:
        print(f"Could not index Kaggle images: {exc}")
        return 1

    overlaps = find_patient_split_overlaps(records)
    print(f"Indexed {len(records)} Kaggle images.")
    if overlaps:
        print(f"Patients in multiple source splits: {len(overlaps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
