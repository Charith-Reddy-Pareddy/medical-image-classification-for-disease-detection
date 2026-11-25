"""Write local patient-level split assignments as a CSV manifest."""

import csv
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from chest_xray_audit.kaggle_index import (
    DatasetLayoutError,
    ImageRecord,
    summarize_assigned_splits,
)

_COLUMNS = ("image_path", "patient_id", "label", "source_split", "assigned_split")
_LABEL_DIRS = {0: "NORMAL", 1: "PNEUMONIA"}


def write_patient_split_manifest(
    records: Iterable[ImageRecord],
    output_path: Path,
) -> Path:
    """Write split assignments with paths relative to the Kaggle image root."""
    record_list = list(records)
    summarize_assigned_splits(record_list)

    rows = []
    for record in sorted(
        record_list,
        key=lambda row: (row.source_split, row.label, row.patient_id, row.path.name),
    ):
        split = Path(record.source_split)
        if split.is_absolute() or len(split.parts) != 1 or split.parts[0] in {".", ".."}:
            raise DatasetLayoutError("source_split must be one path segment")

        label_dir = _LABEL_DIRS.get(record.label)
        if label_dir is None:
            raise DatasetLayoutError(f"Unsupported Kaggle label: {record.label}")

        image_path = PurePosixPath(
            record.source_split,
            label_dir,
            record.path.name,
        ).as_posix()
        rows.append(
            {
                "image_path": image_path,
                "patient_id": record.patient_id,
                "label": record.label,
                "source_split": record.source_split,
                "assigned_split": record.assigned_split,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return output_path
