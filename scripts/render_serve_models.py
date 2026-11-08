"""Build reports/serve_models.md from the Phase 6a files plus the hand-written notes.

Usage: uv run python scripts/render_serve_models.py     (reads no data, runs no model)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from defectlens.eval.threshold_runs import with_notes
from defectlens.export.serve_report import render_serve_models
from defectlens.serve.config import load_settings

CONSISTENCY = Path("reports/serve_threshold_consistency.json")
BENCH = Path("reports/benchmark.json")
REPORT = Path("reports/serve_models.md")
NOTES = Path("reports/serve_models_notes.md")


def main() -> int:
    text = render_serve_models(
        load_settings(), json.loads(CONSISTENCY.read_text()), json.loads(BENCH.read_text())["rows"]
    )
    REPORT.write_text(with_notes(text, NOTES))
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
