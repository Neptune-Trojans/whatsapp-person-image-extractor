"""Pure-numpy face-embedding comparison helpers.

This module deliberately has no InsightFace dependency so it can be unit-tested
without downloading a model. It operates on objects that expose a
``normed_embedding`` (an L2-normalized 512-d vector) and a ``det_score`` float —
which is exactly the shape of an InsightFace ``Face``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    For already L2-normalized inputs this is just the dot product; we normalize
    defensively so the function is correct for arbitrary vectors too.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _best_face(faces: Sequence) -> object | None:
    """Return the face with the highest detection score, or None if empty."""
    if not faces:
        return None
    return max(faces, key=lambda f: float(f.det_score))


def build_reference_embedding(reference_faces: Sequence) -> np.ndarray:
    """Combine reference faces into a single L2-normalized embedding.

    Expects one representative face per reference image (e.g. the highest
    detection-score face). Averages their embeddings and re-normalizes so the
    result is comparable via :func:`cosine_similarity`.

    Raises:
        ValueError: if ``reference_faces`` is empty.
    """
    if not reference_faces:
        raise ValueError("Cannot build a reference embedding from zero faces")

    embeddings = np.stack(
        [np.asarray(f.normed_embedding, dtype=np.float32) for f in reference_faces]
    )
    mean = embeddings.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        # Degenerate case (embeddings cancelled out); return the zero vector.
        return mean
    return mean / norm


def best_score(reference_embedding: np.ndarray, faces: Sequence) -> float:
    """Best cosine similarity between the reference and any face in an image.

    Returns 0.0 when the image contains no faces.
    """
    if not faces:
        return 0.0
    return max(
        cosine_similarity(reference_embedding, f.normed_embedding) for f in faces
    )
