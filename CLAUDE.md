# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, offline tool that scans a folder of images (typically an unzipped WhatsApp chat export), detects faces, and copies out every image containing a target person identified by a few reference photos. Matching is done with InsightFace ArcFace embeddings compared by cosine similarity.

## Setup & commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # installs the package editable (-e .) + deps
```

The package is installed **editable**, which is what lets `scripts/*.py` do `import face_finder` with no `sys.path` hacks. After cloning, you must `pip install -r requirements.txt` (or `pip install -e .`) before any script runs.

Entry points (all run from the project root):

```bash
python scripts/serve.py [--port 5000]        # local web UI at http://127.0.0.1:5000
python scripts/find_person.py -r data/reference -m "data/<chat>" -o output/x -t 0.35
python scripts/extract_all.py                # unzip every *.zip under data/ into sibling folders
python scripts/visualize_detections.py --image path/to/img.jpg   # debug: draw detected boxes
```

The InsightFace `buffalo_l` model auto-downloads to `~/.insightface` on first face detection (not at import time).

## Tests

`tests/` currently contains only `__init__.py` — there is no test suite or pytest config yet. The `matcher` module is deliberately written to be unit-testable without a model (see below); new tests would go in `tests/` and run with `pytest`.

## Architecture

Library code lives in `src/face_finder/`; the four `scripts/` are thin CLI/web entry points that call into it. `pipeline.find_person()` is the single orchestration seam everything routes through.

Key design boundary — **InsightFace/OpenCV are isolated to two modules** so the rest of the package (and tests) stay cheap and importable:
- `faces/analyzer.py` — `FaceAnalyzer`, the *only* code that imports InsightFace. Loads the model lazily on first `embed_image()`. Reads files via `read_bytes()` + `cv2.imdecode` (not `cv2.imread`) so non-ASCII filenames (Hebrew/Arabic) decode reliably.
- `faces/matcher.py` — pure-numpy embedding math (cosine similarity, averaging reference embeddings, scoring). No InsightFace dependency by design.

`pipeline.py` owns all I/O and orchestration: builds one averaged reference embedding from the reference images, scans every image, writes `matches.csv`, copies matches. Two record types matter:
- A **Detection** is emitted per detected face (and one face-less row per image with no faces) → every scanned image appears in the manifest.
- A **Match** is emitted only when an image's *best* face similarity ≥ threshold → that image is copied to `out_dir`.

`embed_image()` has a three-way return contract the pipeline depends on: `None` = unreadable/undecodable (counts as *skipped*), `[]` = decoded but no faces (counts as *scanned*, face-less manifest row), non-empty list = faces found.

`web/app.py` — single-page Flask UI over `find_person()`. `JobManager` runs one background job at a time behind a lock and exposes live progress via the pipeline's `on_progress(done, total, matches)` callback, polled by `/api/status`. No matching logic lives here. Bound to 127.0.0.1, single-user, no auth.

`config.py` — all defaults (model name, `DET_SIZE`, `DEFAULT_THRESHOLD=0.35`, `IMAGE_EXTS`, `PROVIDERS`) and the well-known `DATA_DIR`/`OUTPUT_DIR` paths. `PROVIDERS` is CPU-only by default; prepend a CoreML provider for Apple Silicon GPU.

## Conventions

- `data/` and `output/` are gitignored — they hold private photos. Never commit their contents.
- The pipeline skips re-scanning its own `out_dir` when it lives inside `media_dir`.
- `copy_into` overwrites existing files of the same name (last match wins on name collision).
