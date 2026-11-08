"""Build reports/unseen_followup.md: the EXPLORATORY follow-up to Phase 4b (trained on vs held out).

Usage: uv run python scripts/render_unseen_followup.py

Reads every `reports/unseen/pcb1_*.json` (reads no data, runs no model) and writes
`reports/unseen_followup.md` followed by the hand-written notes from
`reports/unseen_followup_notes.md` (a TODO placeholder until they exist).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from defectlens.eval.threshold_runs import with_notes
from defectlens.eval.unseen import check_same_recipe
from defectlens.eval.unseen_followup import render_followup

RESULTS = Path("reports/unseen")
REPORT = Path("reports/unseen_followup.md")
NOTES = Path("reports/unseen_followup_notes.md")


def main() -> int:
    results = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("pcb1_*.json"))]
    check_same_recipe(results)
    runs = {str(r["holdout"]): r for r in results}
    REPORT.write_text(with_notes(render_followup(runs), NOTES))
    print(f"wrote {REPORT} from {len(results)} result file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
