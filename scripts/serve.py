"""Phase 6a: run the API.

Usage:
  uv run python scripts/serve.py                         # configs/serve/default.yaml, port 8000
  uv run python scripts/serve.py --port 8080 --preload steel
  curl -F image=@tests/fixtures/sample.jpg "localhost:8000/inspect?product=steel"

Models are loaded on first use (or `--preload`), under the device-memory budget in the config.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from defectlens.serve.config import CONFIG_ENV, DEFAULT_CONFIG


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside a container")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()
    os.environ[CONFIG_ENV] = args.config
    uvicorn.run(
        "defectlens.serve.app:create_app", factory=True, host=args.host, port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
