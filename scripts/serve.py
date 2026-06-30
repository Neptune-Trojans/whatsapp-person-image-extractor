#!/usr/bin/env python3
"""Launch the local web UI for the face-finding pipeline.

    python scripts/serve.py            # http://127.0.0.1:5000
    python scripts/serve.py --port 8000

Open the printed URL in a browser to pick folders and run.
"""

from __future__ import annotations

import argparse

from face_finder.web import create_app

HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})."
    )
    args = parser.parse_args(argv)

    app = create_app()
    print(f"Face Finder UI running at http://{HOST}:{args.port}  (Ctrl-C to stop)")
    # threaded=True so /api/status keeps responding while a scan runs.
    app.run(host=HOST, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
