"""Helpers for locating media files to scan."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..config import IMAGE_EXTS


def iter_images(root: Path) -> Iterator[Path]:
    """Yield image files under ``root`` (recursively), sorted for determinism.

    Only files whose suffix is in :data:`face_finder.config.IMAGE_EXTS` are
    returned. The comparison is case-insensitive.
    """
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path
