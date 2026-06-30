"""Core face-finding pipeline: load references, scan media, copy matches.

Orchestrates :mod:`face_finder.media`, :mod:`face_finder.faces.analyzer`, and
:mod:`face_finder.matcher`, and owns all I/O (copying, the manifest CSV, logging
and the run summary).
"""

from __future__ import annotations

import csv
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import DEFAULT_THRESHOLD
from .faces import FaceAnalyzer, matcher
from .utils import iter_images

log = logging.getLogger(__name__)

MANIFEST_NAME = "matches.csv"


@dataclass(frozen=True)
class Match:
    """A media image whose best face matched the reference person."""

    source: Path        # original image path under media_dir
    similarity: float   # best cosine similarity vs reference (0..1)
    num_faces: int      # faces detected in this image
    output: Path        # where the copy was written under out_dir


@dataclass(frozen=True)
class Detection:
    """One detected face on a scanned image (or a face-less image).

    ``box`` and ``similarity`` are ``None`` for images where no face was
    detected, so every scanned image appears in the manifest.
    """

    source: Path                                          # image path under media_dir
    box: tuple[float, float, float, float] | None         # (x_min, y_min, x_max, y_max)
    similarity: float | None                              # cosine similarity for this face


@dataclass
class PipelineResult:
    """Outcome of a pipeline run."""

    matches: list[Match]
    scanned: int          # images successfully analyzed
    skipped: int          # unreadable/undecodable images
    reference_faces: int  # faces used to build the reference embedding
    manifest: Path        # path to the matches CSV


def find_person(
    reference_dir: Path,
    media_dir: Path,
    out_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    analyzer: FaceAnalyzer | None = None,
    progress: bool = True,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> PipelineResult:
    """Find images of the reference person within ``media_dir``.

    Args:
        reference_dir: Folder of reference image(s) of the target person.
        media_dir: Folder of images to scan (searched recursively).
        out_dir: Folder to copy matching images into; the manifest CSV is
            written here too. Created if it does not exist.
        threshold: Minimum cosine similarity for a match.
        analyzer: Optional pre-built analyzer (injected in tests). When ``None``
            a default :class:`FaceAnalyzer` is constructed.
        progress: Whether to show a tqdm progress bar while scanning.
        on_progress: Optional callback invoked after each scanned image with
            ``(done, total, matches_so_far)`` — used by the web UI for live
            progress.

    Returns:
        A :class:`PipelineResult` summarising the run.

    Raises:
        ValueError: if no reference images or no reference faces are found.
    """
    reference_dir = Path(reference_dir)
    media_dir = Path(media_dir)
    out_dir = Path(out_dir)

    if analyzer is None:
        analyzer = FaceAnalyzer()

    reference_embedding, reference_faces = _build_reference(reference_dir, analyzer)

    out_dir.mkdir(parents=True, exist_ok=True)
    matches, detections, scanned, skipped = _scan(
        media_dir, out_dir, analyzer, reference_embedding, threshold, progress,
        on_progress,
    )

    manifest = _write_manifest(out_dir, media_dir, detections)

    log.info(
        "scanned %d image(s), skipped %d, found %d match(es) >= %.2f -> %s",
        scanned,
        skipped,
        len(matches),
        threshold,
        out_dir,
    )
    return PipelineResult(
        matches=matches,
        scanned=scanned,
        skipped=skipped,
        reference_faces=reference_faces,
        manifest=manifest,
    )


def _build_reference(
    reference_dir: Path, analyzer: FaceAnalyzer
) -> tuple[np.ndarray, int]:
    """Build a single reference embedding from the reference images."""
    reference_images = list(iter_images(reference_dir))
    if not reference_images:
        raise ValueError(f"No reference images found in {reference_dir}")

    representative_faces = []
    for path in reference_images:
        faces = analyzer.embed_image(path)
        if not faces:  # None (unreadable) or [] (no faces)
            log.warning("No usable face in reference image %s", path)
            continue
        if len(faces) > 1:
            log.warning(
                "Reference image %s has %d faces; using the highest-score one",
                path,
                len(faces),
            )
        representative_faces.append(matcher._best_face(faces))

    if not representative_faces:
        raise ValueError(
            f"No faces detected in any reference image under {reference_dir}"
        )

    embedding = matcher.build_reference_embedding(representative_faces)
    log.info("Built reference embedding from %d face(s)", len(representative_faces))
    return embedding, len(representative_faces)


def _scan(
    media_dir: Path,
    out_dir: Path,
    analyzer: FaceAnalyzer,
    reference_embedding: np.ndarray,
    threshold: float,
    progress: bool,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> tuple[list[Match], list[Detection], int, int]:
    """Scan every image under ``media_dir``, recording per-face detections.

    Returns matches (best face >= threshold, copied out), detections (one row
    per detected face, or one face-less row per image with no faces), and the
    scanned / skipped counts.
    """
    matches: list[Match] = []
    detections: list[Detection] = []
    scanned = 0
    skipped = 0

    out_dir_resolved = out_dir.resolve()

    images = list(iter_images(media_dir))
    total = len(images)
    bar = tqdm(
        images, total=total, desc="Scanning", unit="img", disable=not progress
    )
    for done, path in enumerate(bar, start=1):
        # Don't re-scan our own output if out_dir lives inside media_dir.
        if out_dir_resolved in path.resolve().parents:
            pass
        else:
            faces = analyzer.embed_image(path)
            if faces is None:  # decode failure
                skipped += 1
            elif not faces:  # decoded, but no faces
                scanned += 1
                detections.append(Detection(source=path, box=None, similarity=None))
            else:
                scanned += 1
                face_scores = matcher.scores(reference_embedding, faces)
                for face, score in zip(faces, face_scores):
                    x_min, y_min, x_max, y_max = (float(v) for v in face.bbox)
                    detections.append(
                        Detection(
                            source=path,
                            box=(x_min, y_min, x_max, y_max),
                            similarity=score,
                        )
                    )
                best = max(face_scores)
                if best >= threshold:
                    output = _copy_match(path, out_dir)
                    matches.append(
                        Match(
                            source=path,
                            similarity=best,
                            num_faces=len(faces),
                            output=output,
                        )
                    )

        if on_progress is not None:
            on_progress(done, total, len(matches))

    return matches, detections, scanned, skipped


def _copy_match(source: Path, out_dir: Path) -> Path:
    """Copy ``source`` into ``out_dir``, avoiding name collisions."""
    target = out_dir / source.name
    if target.exists():
        stem, suffix = source.stem, source.suffix
        i = 1
        while target.exists():
            target = out_dir / f"{stem}_{i}{suffix}"
            i += 1
    shutil.copy2(source, target)
    return target


def _write_manifest(
    out_dir: Path, media_dir: Path, detections: list[Detection]
) -> Path:
    """Write the per-face manifest CSV.

    One row per detected face (every scanned image included; face-less images
    get a single row with blank box/similarity columns). Rows are in scan
    order, so faces of the same image stay grouped. Always written, even if
    empty.
    """
    manifest = out_dir / MANIFEST_NAME

    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["source_path", "x_min", "x_max", "y_min", "y_max", "similarity"]
        )
        for d in detections:
            try:
                source_path = d.source.relative_to(media_dir)
            except ValueError:
                source_path = d.source

            if d.box is None:
                writer.writerow([str(source_path), "", "", "", "", ""])
                continue

            x_min, y_min, x_max, y_max = d.box
            writer.writerow(
                [
                    str(source_path),
                    f"{x_min:.1f}",
                    f"{x_max:.1f}",
                    f"{y_min:.1f}",
                    f"{y_max:.1f}",
                    f"{d.similarity:.4f}",
                ]
            )

    return manifest
