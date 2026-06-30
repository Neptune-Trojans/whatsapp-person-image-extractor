#!/usr/bin/env python3
"""Run face detection on a single image and save an annotated visualization.

Detects faces with the InsightFace analyzer, draws their boxes / detection
scores / landmarks, and writes the result under ``output/visualizations/``
(or a path you choose). Run from the project root:

    python scripts/visualize_detections.py --image "data/<chat>/00002340-PHOTO-....jpg"
    python scripts/visualize_detections.py --image path/to/img.jpg --out output/vis/out.jpg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from face_finder.config import OUTPUT_DIR, PROJECT_ROOT
from face_finder.faces import FaceAnalyzer
from face_finder.visualization import draw_faces

DEFAULT_OUT_DIR = OUTPUT_DIR / "visualizations"

log = logging.getLogger("visualize_detections")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path for the annotated image. "
        f"Defaults to {DEFAULT_OUT_DIR.relative_to(PROJECT_ROOT)}/<name>_detected<ext>.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    image_path: Path = args.image
    if not image_path.is_file():
        log.error("Image not found: %s", image_path)
        return 1

    out_path: Path = args.out or (
        DEFAULT_OUT_DIR / f"{image_path.stem}_detected{image_path.suffix}"
    )

    analyzer = FaceAnalyzer()
    faces = analyzer.embed_image(image_path)
    if faces is None:
        log.error("Could not read or decode image: %s", image_path)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    draw_faces(image_path, faces, output_path=out_path)

    log.info("Detected %d face(s) in %s", len(faces), image_path.name)
    log.info("Wrote visualization -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
