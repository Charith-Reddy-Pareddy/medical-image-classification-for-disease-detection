"""Read the repository's dataset manifest."""

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs" / "datasets.toml"


def load_dataset_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load dataset definitions from a TOML file."""
    with path.open("rb") as stream:
        return tomllib.load(stream)
