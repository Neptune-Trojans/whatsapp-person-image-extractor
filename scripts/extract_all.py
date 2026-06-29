#!/usr/bin/env python3
"""Extract every zip file under ``data/`` in place.

Each archive is extracted into its own subfolder under ``data/`` named after
the zip (without the .zip suffix). Run from the project root:

    python scripts/extract_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from face_finder.utils import extract_zip

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"


def main() -> int:
    if not DATA_DIR.is_dir():
        print(f"Data directory not found: {DATA_DIR}", file=sys.stderr)
        return 1

    zips = sorted(DATA_DIR.rglob("*.zip"))
    if not zips:
        print(f"No zip files found under {DATA_DIR}")
        return 0

    for zip_path in zips:
        dest = zip_path.with_suffix("")
        print(f"Extracting {zip_path.relative_to(PROJECT_ROOT)} -> "
              f"{dest.relative_to(PROJECT_ROOT)}")
        extract_zip(zip_path, dest)

    print(f"Done. Extracted {len(zips)} archive(s) under {DATA_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
