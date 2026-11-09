"""Phase 6c: rewrite the generated result tables of README.md from reports/.

Usage:
  uv run python scripts/build_report_tables.py            # rewrite README.md in place
  uv run python scripts/build_report_tables.py --check    # exit 1 if README.md is out of date
  uv run python scripts/build_report_tables.py --print    # print the tables, touch nothing

Reads only files under reports/ (no MLflow, no data, no model). A table that a report no longer has
makes the script fail instead of leaving a stale one in the README.
See defectlens.eval.readme_tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from defectlens.eval.readme_tables import ReportError, build_blocks, unmarked_names, update_readme

README = Path("README.md")
REPORTS = Path("reports")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="only compare; exit 1 if out of date")
    mode.add_argument("--print", action="store_true", help="print the tables, write nothing")
    args = ap.parse_args()

    try:
        blocks = build_blocks(REPORTS)
        if args.print:
            for name, body in blocks.items():
                print(f"<!-- generated:{name} -->\n{body}\n<!-- /generated:{name} -->\n")
            return 0
        old = README.read_text()
        stray = unmarked_names(old)
        if stray:
            print(f"README.md has marker pairs no block fills: {stray}", file=sys.stderr)
            return 1
        new = update_readme(old, blocks)
    except ReportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.check:
        if new != old:
            print("README.md is out of date: run scripts/build_report_tables.py", file=sys.stderr)
            return 1
        print("README.md tables are up to date")
        return 0
    README.write_text(new)
    print(f"wrote {README} ({len(blocks)} generated tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
