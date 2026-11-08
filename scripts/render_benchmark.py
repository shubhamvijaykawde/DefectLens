"""Build reports/benchmark.md from the Phase 5a result files plus the hand-written notes.

Usage: uv run python scripts/render_benchmark.py     (reads no data, runs no model)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from defectlens.eval.threshold_runs import with_notes
from defectlens.export.report import render_benchmark

EXPORT = Path("reports/onnx_export.json")
VERIFY = Path("reports/onnx_verification.json")
BENCH = Path("reports/benchmark.json")
REPORT = Path("reports/benchmark.md")
NOTES = Path("reports/benchmark_notes.md")


def load(path: Path) -> object | None:
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> int:
    text = render_benchmark(load(EXPORT), load(VERIFY), load(BENCH))  # type: ignore[arg-type]
    REPORT.write_text(with_notes(text, NOTES))
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
