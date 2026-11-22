"""Read the repository's dataset manifest."""

import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs" / "datasets.toml"
_OPTIONAL_PATH_FIELDS = ("metadata_file", "reports_dir")
_DATA_PATH_FIELDS = ("image_dir", *_OPTIONAL_PATH_FIELDS)
_ROLES = {"train", "external_validation"}
_SPLITS = {"train", "val", "test"}


class ManifestError(ValueError):
    """Raised when a dataset manifest does not match the supported schema."""


def _require_text(record: dict[str, Any], key: str, name: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name}.{key} must be a non-empty string")
    return value


def validate_dataset_manifest(manifest: Any) -> None:
    """Check the manifest's required fields and repository-relative paths."""
    if not isinstance(manifest, dict):
        raise ManifestError("manifest must be a TOML table")

    version = manifest.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ManifestError("schema_version must be 1")

    data_root = _require_text(manifest, "data_root", "manifest")
    _check_relative_path(data_root, "manifest.data_root")
    _validate_patient_split_config(manifest)

    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ManifestError("datasets must be a non-empty table")

    for name, dataset in datasets.items():
        if not isinstance(name, str) or not name.strip():
            raise ManifestError("dataset names must be non-empty strings")
        if not isinstance(dataset, dict):
            raise ManifestError(f"datasets.{name} must be a table")

        role = _require_text(dataset, "role", f"datasets.{name}")
        if role not in _ROLES:
            raise ManifestError(f"datasets.{name}.role is not supported")

        for key in ("image_dir", "group_unit", "group_key"):
            value = _require_text(dataset, key, f"datasets.{name}")
            if key == "image_dir":
                _check_relative_path(value, f"datasets.{name}.{key}")

        for key in _OPTIONAL_PATH_FIELDS:
            if key in dataset:
                value = _require_text(dataset, key, f"datasets.{name}")
                _check_relative_path(value, f"datasets.{name}.{key}")

        source_splits = dataset.get("source_splits")
        if source_splits is not None and (
            not isinstance(source_splits, list)
            or not source_splits
            or any(not isinstance(split, str) or not split.strip() for split in source_splits)
        ):
            raise ManifestError(
                f"datasets.{name}.source_splits must be a non-empty list of strings"
            )



def _validate_patient_split_config(manifest: dict[str, Any]) -> None:
    if "patient_split" not in manifest:
        return

    config = manifest["patient_split"]
    if not isinstance(config, dict):
        raise ManifestError("patient_split must be a table")

    ratios = config.get("ratios")
    if not isinstance(ratios, dict) or set(ratios) != _SPLITS:
        raise ManifestError("patient_split.ratios must define train, val, and test")

    for split, ratio in ratios.items():
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or ratio < 0
        ):
            raise ManifestError("patient_split ratios must be finite non-negative numbers")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ManifestError("patient_split ratios must sum to 1")

    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ManifestError("patient_split.seed must be an integer")

    if not isinstance(config.get("stratify_by_label"), bool):
        raise ManifestError("patient_split.stratify_by_label must be a boolean")


def _check_relative_path(value: str, name: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"{name} must stay within the repository")


def resolve_dataset_paths(
    manifest: dict[str, Any],
    *,
    project_root: Path = ROOT,
    data_root: Path | None = None,
) -> dict[str, dict[str, Path]]:
    """Resolve dataset paths relative to the manifest's or supplied data root."""
    validate_dataset_manifest(manifest)

    root = Path(project_root)
    data_dir = Path(data_root) if data_root is not None else Path(manifest["data_root"])
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    data_dir = data_dir.resolve()

    return {
        name: {
            field: (data_dir / dataset[field]).resolve()
            for field in _DATA_PATH_FIELDS
            if field in dataset
        }
        for name, dataset in manifest["datasets"].items()
    }


def find_missing_dataset_inputs(
    manifest: dict[str, Any],
    *,
    project_root: Path = ROOT,
    data_root: Path | None = None,
) -> list[Path]:
    """Return declared files or directories that are absent or the wrong type."""
    dataset_paths = resolve_dataset_paths(manifest, project_root=project_root, data_root=data_root)
    missing = []
    for name, paths in dataset_paths.items():
        for field in _DATA_PATH_FIELDS:
            path = paths.get(field)
            if path is None:
                continue
            if field == "image_dir" and "source_splits" in manifest["datasets"][name]:
                missing.extend(
                    path / split
                    for split in manifest["datasets"][name]["source_splits"]
                    if not (path / split).is_dir()
                )
                continue
            present = path.is_file() if field == "metadata_file" else path.is_dir()
            if not present:
                missing.append(path)
    return missing


def load_dataset_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load dataset definitions from a TOML file."""
    with path.open("rb") as stream:
        manifest = tomllib.load(stream)
    validate_dataset_manifest(manifest)
    return manifest
