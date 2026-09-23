#!/usr/bin/env python3
"""Launcher for the ANTOD Web GUI Dashboard.

Usage:
    python scripts/gui.py
    python scripts/gui.py --port 8080 --no-browser
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from antod.gui.server import start_server  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch the ANTOD Web Dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    start_server(port=args.port, host=args.host, open_browser=not args.no_browser)
