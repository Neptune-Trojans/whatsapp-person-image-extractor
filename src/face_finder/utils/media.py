"""Helpers for locating and copying media files."""

from __future__ import annotations

import shutil
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


def copy_into(source: Path, dest_dir: Path) -> Path:
    """Copy ``source`` into ``dest_dir`` under its original name.

    Any existing file of the same name is overwritten. Returns the destination
    path.
    """
    target = dest_dir / source.name
    shutil.copy2(source, target)
    return target
