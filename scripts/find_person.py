#!/usr/bin/env python3
"""Find images of a reference person within a media folder and copy out matches.

Detects faces in every image under ``--media``, compares them to the person in
``--reference``, and copies matches into ``--out`` along with a ``matches.csv``
manifest. Run from the project root:

    python scripts/find_person.py \\
        --reference data/reference \\
        --media "data/WhatsApp Chat - ..." \\
        --out output/aylon \\
        --threshold 0.35
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from face_finder.config import DEFAULT_THRESHOLD
from face_finder.pipeline import find_person

log = logging.getLogger("find_person")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        required=True,
        help="Folder of reference image(s) of the target person.",
    )
    parser.add_argument(
        "-m",
        "--media",
        type=Path,
        required=True,
        help="Folder of images to scan (searched recursively).",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Folder to copy matching images into; matches.csv is written here.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine-similarity match cutoff (default: {DEFAULT_THRESHOLD}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    if not args.reference.is_dir():
        log.error("Reference directory not found: %s", args.reference)
        return 1
    if not args.media.is_dir():
        log.error("Media directory not found: %s", args.media)
        return 1

    try:
        result = find_person(args.reference, args.media, args.out, args.threshold)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    log.info("Manifest written to %s", result.manifest)
    if result.matches:
        top = sorted(result.matches, key=lambda m: m.similarity, reverse=True)[:5]
        log.info("Top matches:")
        for m in top:
            log.info("  %.4f  %s", m.similarity, m.source.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
