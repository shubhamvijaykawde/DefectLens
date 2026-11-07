"""Re-render reports/thresholds.md from the saved evaluation (no model, no data, no scoring).

Usage: uv run python scripts/render_thresholds.py

Reads `reports/frozen_thresholds.json` (verified) and `reports/threshold_evaluation.json`, checks
that the two belong together, and rewrites `reports/thresholds.md` with the hand-written notes
from `reports/thresholds_notes.md`. Use it after editing the notes or the report layout; it
never evaluates anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from defectlens.eval.threshold_runs import render_report, with_notes
from defectlens.eval.thresholds import ProtocolError, file_sha256, read_frozen

FROZEN = Path("reports/frozen_thresholds.json")
RESULTS = Path("reports/threshold_evaluation.json")


def main() -> int:
    entries, meta = read_frozen(FROZEN)
    sha = file_sha256(FROZEN)
    saved = json.loads(RESULTS.read_text())
    if saved["frozen_file_sha256"] != sha:
        msg = f"{RESULTS} was produced from a different frozen file ({saved['frozen_file_sha256']})"
        raise ProtocolError(msg)
    text = render_report(entries, saved["rows"], saved["context"], sha, meta)
    Path("reports/thresholds.md").write_text(with_notes(text, Path("reports/thresholds_notes.md")))
    print("wrote reports/thresholds.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
