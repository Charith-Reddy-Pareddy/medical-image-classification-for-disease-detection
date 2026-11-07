"""Index the Kaggle chest X-ray folder structure without loading image pixels."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chest_xray_audit.data_manifest import (
    ROOT,
    ManifestError,
    resolve_dataset_paths,
    validate_dataset_manifest,
)

_LABELS = {"NORMAL": 0, "PNEUMONIA": 1}
_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
_PATIENT_ID = re.compile(r"^(person\d+)_", re.IGNORECASE)


class DatasetLayoutError(ValueError):
    """Raised when an image filename cannot provide its required patient key."""


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    source_split: str
    patient_id: str


def build_kaggle_index(
    manifest: dict[str, Any],
    *,
    project_root: Path = ROOT,
    data_root: Path | None = None,
) -> list[ImageRecord]:
    """Build sorted image records from the manifest's Kaggle class folders."""
    validate_dataset_manifest(manifest)
    dataset = manifest["datasets"].get("kaggle")
    if not isinstance(dataset, dict):
        raise ManifestError("datasets.kaggle must be a table")
    if dataset.get("label_source") != "class_directory":
        raise ManifestError("datasets.kaggle.label_source must be class_directory")
    if dataset.get("group_unit") != "patient" or dataset.get("group_key") != "filename_patient_key":
        raise ManifestError("datasets.kaggle must use filename-based patient grouping")

    splits = dataset.get("source_splits")
    if not isinstance(splits, list) or not splits:
        raise ManifestError("datasets.kaggle.source_splits must be a non-empty list")

    image_dir = resolve_dataset_paths(manifest, project_root=project_root, data_root=data_root)[
        "kaggle"
    ]["image_dir"]
    records = []
    for split in splits:
        for class_name, label in _LABELS.items():
            class_dir = image_dir / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing Kaggle class directory: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                match = _PATIENT_ID.match(path.name)
                if match is None:
                    raise DatasetLayoutError(f"Cannot read patient ID from filename: {path.name}")
                records.append(
                    ImageRecord(
                        path=path,
                        label=label,
                        source_split=split,
                        patient_id=match.group(1).lower(),
                    )
                )
    return records
