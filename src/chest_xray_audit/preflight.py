"""Command-line check for locally available dataset inputs."""

import argparse
from pathlib import Path

from chest_xray_audit.data_manifest import find_missing_dataset_inputs, load_dataset_manifest


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
