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


def read_patient_split_manifest(
    input_path: Path,
    *,
    image_root: Path | None = None,
) -> list[dict[str, str]]:
    """Read and validate rows, optionally checking image files under a root."""
    root = Path(image_root).resolve() if image_root is not None else None
    with Path(input_path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise DatasetLayoutError("Split manifest must include a header row")
        if len(fieldnames) != len(set(fieldnames)):
            raise DatasetLayoutError("Split manifest columns must be unique")

        missing = sorted(set(_COLUMNS) - set(fieldnames))
        if missing:
            raise DatasetLayoutError(
                f"Split manifest is missing columns: {', '.join(missing)}"
            )

        rows = []
        patient_splits = {}
        image_paths = set()
        for line_num, row in enumerate(reader, start=2):
            if None in row:
                raise DatasetLayoutError(
                    f"Split manifest row {line_num} has extra values"
                )

            record = {column: row.get(column) for column in _COLUMNS}
            if any(value is None or not value.strip() for value in record.values()):
                raise DatasetLayoutError(
                    f"Split manifest row {line_num} has empty required values"
                )

            image_path = record["image_path"]
            patient_id = record["patient_id"]
            label = record["label"]
            source_split = record["source_split"]
            assigned_split = record["assigned_split"]

            if source_split not in {"train", "val", "test"}:
                raise DatasetLayoutError(
                    f"Split manifest row {line_num} has an invalid source split"
                )
            if assigned_split not in {"train", "val", "test"}:
                raise DatasetLayoutError(
                    f"Split manifest row {line_num} has an invalid assigned split"
                )
            if label not in {"0", "1"}:
                raise DatasetLayoutError(
                    f"Split manifest row {line_num} has an invalid label"
                )

            path = PurePosixPath(image_path)
            parts = path.parts
            label_dir = _LABEL_DIRS[int(label)]
            if (
                "\\" in image_path
                or path.is_absolute()
                or path.as_posix() != image_path
                or len(parts) != 3
                or any(part in {".", ".."} for part in parts)
                or parts[0] != source_split
                or parts[1] != label_dir
                or path.suffix.lower() not in {".jpeg", ".jpg", ".png"}
            ):
                raise DatasetLayoutError(
                    f"Split manifest row {line_num} has an invalid relative image path"
                )
            if root is not None:
                resolved_path = (root / Path(*parts)).resolve()
                if not resolved_path.is_relative_to(root):
                    raise DatasetLayoutError(
                        f"Split manifest row {line_num} image path escapes image root"
                    )
                if not resolved_path.is_file():
                    raise DatasetLayoutError(
                        f"Split manifest row {line_num} image file not found under image root"
                    )

            if image_path in image_paths:
                raise DatasetLayoutError(
                    f"Split manifest row {line_num} repeats an image path"
                )
            image_paths.add(image_path)

            previous = patient_splits.setdefault(patient_id, assigned_split)
            if previous != assigned_split:
                raise DatasetLayoutError(
                    f"Patient leakage across assigned splits: {patient_id}"
                )
            rows.append(record)

        return rows
