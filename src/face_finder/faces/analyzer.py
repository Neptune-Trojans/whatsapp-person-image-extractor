"""Thin wrapper around InsightFace so the rest of the package stays decoupled.

Only this module imports InsightFace/OpenCV. It loads the model lazily (and the
model is downloaded on first use), so importing the package — and unit-testing
the pure-numpy matcher — stays cheap.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..config import DET_SIZE, MODEL_NAME, PROVIDERS

log = logging.getLogger(__name__)


class FaceAnalyzer:
    """Detects faces and produces embeddings for an image.

    The underlying InsightFace ``FaceAnalysis`` app is constructed lazily on the
    first call to :meth:`embed_image`.
    """

    def __init__(
        self,
        model: str = MODEL_NAME,
        providers: list[str] | None = None,
        det_size: tuple[int, int] = DET_SIZE,
    ) -> None:
        self._model = model
        self._providers = list(providers) if providers is not None else list(PROVIDERS)
        self._det_size = det_size
        self._app = None  # built on first use

    def _ensure_app(self):
        if self._app is None:
            from insightface.app import FaceAnalysis

            log.info("Loading InsightFace model %r (providers=%s)", self._model, self._providers)
            app = FaceAnalysis(name=self._model, providers=self._providers)
            app.prepare(ctx_id=0, det_size=self._det_size)
            self._app = app
        return self._app

    def embed_image(self, path: Path) -> list | None:
        """Detect faces in ``path`` and return them.

        Returns:
            ``None`` if the file cannot be read or decoded;
            ``[]`` if it decoded but contained no faces;
            otherwise the list of detected faces, each exposing
            ``normed_embedding`` (L2-normalized 512-d) and ``det_score``.
        """
        try:
            # Read via bytes + imdecode so non-ASCII filenames (e.g. Hebrew) work
            # reliably, unlike cv2.imread which mishandles them on some platforms.
            data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except OSError as exc:
            log.warning("Could not read %s: %s", path, exc)
            return None

        if img is None:
            log.warning("Could not decode image %s", path)
            return None

        app = self._ensure_app()
        return app.get(img)
