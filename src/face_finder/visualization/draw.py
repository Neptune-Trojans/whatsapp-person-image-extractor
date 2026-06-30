"""Draw detected face boxes (and optional landmarks/labels) onto an image.

Useful for visually inspecting and tuning detection/matching: render the boxes
an analyzer found, optionally annotated with similarity scores, and save or
return the result.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

# OpenCV uses BGR ordering.
DEFAULT_COLOR = (0, 255, 0)   # green boxes
KEYPOINT_COLOR = (0, 0, 255)  # red landmarks
ImageInput = "np.ndarray | str | Path"


def _load_image(image) -> np.ndarray:
    """Return a writable BGR image from an ndarray or a file path.

    Paths are read via bytes + imdecode so non-ASCII filenames (e.g. Hebrew)
    work reliably, consistent with the analyzer.
    """
    if isinstance(image, np.ndarray):
        return image.copy()
    data = np.frombuffer(Path(image).read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {image}")
    return img


def _save_image(img: np.ndarray, output_path) -> Path:
    """Write ``img`` to ``output_path`` (encoder chosen by the file suffix)."""
    path = Path(output_path)
    suffix = path.suffix or ".jpg"
    ok, buf = cv2.imencode(suffix, img)
    if not ok:
        raise ValueError(f"Could not encode image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())
    return path


def draw_detections(
    image,
    boxes: Iterable[Sequence[float]],
    *,
    labels: Sequence[str] | None = None,
    keypoints: Iterable[Sequence[Sequence[float]]] | None = None,
    color: tuple[int, int, int] = DEFAULT_COLOR,
    thickness: int = 2,
    font_scale: float = 0.6,
    output_path=None,
) -> np.ndarray:
    """Draw bounding boxes (and optional labels/landmarks) on an image.

    Args:
        image: A BGR ndarray, or a path to an image file.
        boxes: Iterable of ``(x1, y1, x2, y2)`` boxes in pixel coordinates.
        labels: Optional text per box (e.g. similarity scores). Must match the
            number of boxes when given.
        keypoints: Optional iterable of per-face landmark arrays (each a list of
            ``(x, y)`` points), e.g. InsightFace ``face.kps``.
        color: Box (and label background) colour in BGR.
        thickness: Box line thickness in pixels.
        font_scale: Label font scale.
        output_path: If given, the annotated image is also written here.

    Returns:
        The annotated image as a BGR ndarray.
    """
    img = _load_image(image)
    boxes = [tuple(int(round(v)) for v in box) for box in boxes]

    if labels is not None:
        labels = list(labels)
        if len(labels) != len(boxes):
            raise ValueError(
                f"labels ({len(labels)}) must match number of boxes ({len(boxes)})"
            )

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if labels is not None:
            text = labels[i]
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            # Filled background above the box for legibility.
            top = max(y1 - th - baseline - 2, 0)
            cv2.rectangle(img, (x1, top), (x1 + tw + 2, top + th + baseline + 2), color, -1)
            cv2.putText(
                img,
                text,
                (x1 + 1, top + th + baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    if keypoints is not None:
        for kps in keypoints:
            for x, y in kps:
                cv2.circle(img, (int(round(x)), int(round(y))), 2, KEYPOINT_COLOR, -1)

    if output_path is not None:
        _save_image(img, output_path)

    return img


def draw_faces(
    image,
    faces: Sequence,
    *,
    show_score: bool = True,
    show_keypoints: bool = True,
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper to draw InsightFace ``Face`` objects.

    Pulls ``bbox`` (and optionally ``det_score`` as a label and ``kps`` as
    landmarks) off each face and forwards to :func:`draw_detections`.
    """
    boxes = [f.bbox for f in faces]
    labels = [f"{f.det_score:.2f}" for f in faces] if show_score else None
    keypoints = None
    if show_keypoints and all(getattr(f, "kps", None) is not None for f in faces):
        keypoints = [f.kps for f in faces]
    return draw_detections(
        image, boxes, labels=labels, keypoints=keypoints, **kwargs
    )
