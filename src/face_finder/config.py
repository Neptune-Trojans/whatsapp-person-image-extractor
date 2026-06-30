"""Shared configuration defaults for the face-finding pipeline."""

from __future__ import annotations

from pathlib import Path

# Well-known repository directories. This file lives at
# src/face_finder/config.py, so the repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# InsightFace model bundle (detection + ArcFace recognition).
# Auto-downloads to ~/.insightface on first use.
MODEL_NAME = "buffalo_l"

# Detection input size passed to FaceAnalysis.prepare().
DET_SIZE = (640, 640)

# Default cosine-similarity threshold for ArcFace normed embeddings.
# A candidate image matches when its best face similarity >= this value.
DEFAULT_THRESHOLD = 0.35

# ONNX Runtime execution providers. CPU is the reliable default; CoreML can be
# prepended on Apple Silicon for GPU acceleration.
PROVIDERS = ["CPUExecutionProvider"]

# Image file extensions the pipeline will scan (lower-case, with leading dot).
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
