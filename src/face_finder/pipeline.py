"""Core face-finding pipeline: load references, scan media, copy matches.

Orchestrates :mod:`face_finder.media`, :mod:`face_finder.faces.analyzer`, and
:mod:`face_finder.matcher`, and owns all I/O (copying, the manifest CSV, logging
and the run summary).
"""

from __future__ import annotations

import csv
import logging
import shutil
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
        progress: Whether to show a progress bar while scanning.

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
    matches, scanned, skipped = _scan(
        media_dir, out_dir, analyzer, reference_embedding, threshold, progress
    )

    manifest = _write_manifest(out_dir, media_dir, matches)

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
) -> tuple[list[Match], int, int]:
    """Scan every image under ``media_dir`` and copy out the matches."""
    matches: list[Match] = []
    scanned = 0
    skipped = 0

    out_dir_resolved = out_dir.resolve()

    for path in tqdm(
        iter_images(media_dir), desc="Scanning", unit="img", disable=not progress
    ):
        # Don't re-scan our own output if out_dir lives inside media_dir.
        if out_dir_resolved in path.resolve().parents:
            continue

        faces = analyzer.embed_image(path)
        if faces is None:  # decode failure
            skipped += 1
            continue

        scanned += 1
        score = matcher.best_score(reference_embedding, faces)
        if score >= threshold:
            output = _copy_match(path, out_dir)
            matches.append(
                Match(
                    source=path,
                    similarity=score,
                    num_faces=len(faces),
                    output=output,
                )
            )

    return matches, scanned, skipped


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


def _write_manifest(out_dir: Path, media_dir: Path, matches: list[Match]) -> Path:
    """Write matches.csv (best matches first); always written, even if empty."""
    manifest = out_dir / MANIFEST_NAME
    rows = sorted(matches, key=lambda m: m.similarity, reverse=True)

    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["similarity", "num_faces", "source_path", "output_file"])
        for m in rows:
            try:
                source_path = m.source.relative_to(media_dir)
            except ValueError:
                source_path = m.source
            writer.writerow(
                [f"{m.similarity:.4f}", m.num_faces, str(source_path), m.output.name]
            )

    return manifest
