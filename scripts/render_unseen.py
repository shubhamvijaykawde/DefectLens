"""Build reports/unseen_defects.md from the result files of `scripts/run_unseen.py`.

Usage: uv run python scripts/render_unseen.py

Reads every `reports/unseen/pcb1_*.json` (reads no data and runs no model), refuses to mix runs
made with different recipes, and writes `reports/unseen_defects.md` followed by the hand-written
notes from `reports/unseen_defects_notes.md` (a TODO placeholder until they exist). With no
result files it writes the protocol and a TODO, so the report always exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from defectlens.eval.threshold_runs import with_notes
from defectlens.eval.unseen import check_same_recipe, render_report

RESULTS = Path("reports/unseen")
REPORT = Path("reports/unseen_defects.md")
NOTES = Path("reports/unseen_defects_notes.md")


def main() -> int:
    results = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("pcb1_*.json"))]
    check_same_recipe(results)
    REPORT.write_text(with_notes(render_report(results), NOTES))
    print(f"wrote {REPORT} from {len(results)} result file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
