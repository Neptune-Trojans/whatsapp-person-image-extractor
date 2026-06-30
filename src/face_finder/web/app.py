"""A small local Flask UI for running the face-finding pipeline.

Serves a single page to pick the media / output / reference folders, start a
run, and poll live progress. Thin layer over :func:`pipeline.find_person` — no
matching logic lives here. Bind to 127.0.0.1 only; this is a single-user tool
with no auth and a single job at a time.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from ..config import DATA_DIR, DEFAULT_THRESHOLD, OUTPUT_DIR, PROJECT_ROOT
from ..pipeline import find_person


@dataclass
class JobState:
    """Snapshot of the (single) current/last run, shared with the UI."""

    state: str = "idle"          # idle | running | done | error
    phase: str = ""              # human-readable current phase
    done: int = 0                # images processed
    total: int = 0               # images to process
    matches: int = 0             # matches found so far
    message: str = ""            # summary (done) or error (error)
    manifest: str | None = None  # path to matches.csv when done


class JobManager:
    """Owns the single background job and its state behind a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = JobState()
        self._thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        with self._lock:
            return asdict(self._state)

    def is_running(self) -> bool:
        with self._lock:
            return self._state.state == "running"

    def start(self, media: Path, output: Path, reference: Path, threshold: float) -> None:
        with self._lock:
            self._state = JobState(state="running", phase="Loading model / preparing reference…")
        self._thread = threading.Thread(
            target=self._run, args=(media, output, reference, threshold), daemon=True
        )
        self._thread.start()

    def _set(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._state, key, value)

    def _run(self, media: Path, output: Path, reference: Path, threshold: float) -> None:
        def on_progress(done: int, total: int, matches: int) -> None:
            self._set(phase="Scanning…", done=done, total=total, matches=matches)

        try:
            result = find_person(
                reference, media, output, threshold,
                progress=False, on_progress=on_progress,
            )
            self._set(
                state="done",
                phase="Done",
                matches=len(result.matches),
                message=(
                    f"Scanned {result.scanned} image(s), skipped {result.skipped}, "
                    f"found {len(result.matches)} match(es)."
                ),
                manifest=str(result.manifest),
            )
        except Exception as exc:  # ValueError (no ref faces) or anything unexpected
            self._set(state="error", phase="Error", message=str(exc))


def _safe_dir(raw: str | None) -> Path:
    """Resolve ``raw`` to an existing directory, falling back to PROJECT_ROOT."""
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
    return PROJECT_ROOT


def create_app() -> Flask:
    app = Flask(__name__)
    jobs = JobManager()

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            default_media=str(DATA_DIR),
            default_output=str(OUTPUT_DIR),
            default_reference=str(DATA_DIR / "reference"),
            default_threshold=DEFAULT_THRESHOLD,
        )

    @app.get("/api/browse")
    def browse():
        current = _safe_dir(request.args.get("path"))
        dirs = sorted(
            p.name for p in current.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        parent = None if current.parent == current else str(current.parent)
        return jsonify(path=str(current), parent=parent, dirs=dirs)

    @app.post("/api/run")
    def run():
        data = request.get_json(silent=True) or {}
        media = Path(str(data.get("media", ""))).expanduser()
        output = Path(str(data.get("output", ""))).expanduser()
        reference = Path(str(data.get("reference", ""))).expanduser()

        try:
            threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
        except (TypeError, ValueError):
            return jsonify(error="Threshold must be a number."), 400

        if not media.is_dir():
            return jsonify(error=f"Media folder not found: {media}"), 400
        if not reference.is_dir():
            return jsonify(error=f"Reference folder not found: {reference}"), 400
        if jobs.is_running():
            return jsonify(error="A run is already in progress."), 409

        jobs.start(media, output, reference, threshold)
        return jsonify(started=True)

    @app.get("/api/status")
    def status():
        return jsonify(jobs.snapshot())

    return app
