"""Index the Kaggle chest X-ray folder structure without loading image pixels."""

import math
import random
import re
from collections.abc import Iterable, Mapping
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
_DEFAULT_SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


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


def find_patient_split_overlaps(records: Iterable[ImageRecord]) -> dict[str, tuple[str, ...]]:
    """Report patient IDs present in more than one original source split."""
    patient_splits: dict[str, set[str]] = {}
    for record in records:
        patient_splits.setdefault(record.patient_id, set()).add(record.source_split)
    return {
        patient_id: tuple(sorted(splits))
        for patient_id, splits in sorted(patient_splits.items())
        if len(splits) > 1
    }


def group_records_by_patient(
    records: Iterable[ImageRecord],
) -> dict[str, tuple[ImageRecord, ...]]:
    """Keep each patient's image records together for later split assignment."""
    groups: dict[str, list[ImageRecord]] = {}
    for record in records:
        if not record.patient_id:
            raise DatasetLayoutError("patient_id is required for patient grouping")
        groups.setdefault(record.patient_id, []).append(record)
    return {patient_id: tuple(group) for patient_id, group in sorted(groups.items())}


def assign_patient_splits(
    records: Iterable[ImageRecord],
    ratios: Mapping[str, float] | None = None,
    *,
    seed: int = 42,
    stratify_by_label: bool = False,
) -> dict[str, str]:
    """Assign each patient once, optionally stratifying groups by label profile."""
    split_ratios = dict(_DEFAULT_SPLIT_RATIOS if ratios is None else ratios)
    if not split_ratios:
        raise ValueError("at least one split ratio is required")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(stratify_by_label, bool):
        raise ValueError("stratify_by_label must be a boolean")

    for split, ratio in split_ratios.items():
        if not isinstance(split, str) or not split.strip():
            raise ValueError("split names must be non-empty strings")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError("split ratios must be numbers")
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError("split ratios must be finite and non-negative")
    if not math.isclose(sum(split_ratios.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")

    groups = group_records_by_patient(records)
    if not groups:
        return {}

    rng = random.Random(seed)
    if stratify_by_label:
        strata: dict[tuple[int, ...], list[str]] = {}
        for patient_id, group in groups.items():
            labels = tuple(sorted({record.label for record in group}))
            if not labels or any(label not in (0, 1) for label in labels):
                raise ValueError("label stratification requires binary labels")
            strata.setdefault(labels, []).append(patient_id)
        assignments = {}
        for labels in sorted(strata):
            assignments.update(_assign_patient_ids(strata[labels], split_ratios, rng))
        return dict(sorted(assignments.items()))

    return _assign_patient_ids(list(groups), split_ratios, rng)


def _assign_patient_ids(
    patient_ids: list[str],
    ratios: Mapping[str, float],
    rng: random.Random,
) -> dict[str, str]:
    rng.shuffle(patient_ids)
    targets = [len(patient_ids) * ratio for ratio in ratios.values()]
    counts = [math.floor(target) for target in targets]
    remainder_order = sorted(
        range(len(targets)),
        key=lambda index: (-(targets[index] - counts[index]), index),
    )
    for index in remainder_order[: len(patient_ids) - sum(counts)]:
        counts[index] += 1

    assignments = {}
    start = 0
    for (split, _ratio), count in zip(ratios.items(), counts):
        for patient_id in patient_ids[start : start + count]:
            assignments[patient_id] = split
        start += count
    return assignments
