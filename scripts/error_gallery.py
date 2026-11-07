"""Phase 4a, step 3: error gallery at the frozen thresholds (run after evaluate_thresholds.py).

Usage:
  uv run python scripts/error_gallery.py top_k=20            # optional: ratio=10

Reads the evaluation splits at the frozen thresholds and writes `reports/error_gallery.md` and
`reports/figures/error_gallery_*.png`. It changes no threshold. Hand-written notes live in
`reports/error_gallery_notes.md` and are appended on every run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from defectlens.eval.gallery_run import build_gallery


def main() -> int:
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    build_gallery(
        Path("reports/frozen_thresholds.json"), float(args.get("ratio", "10")),
        int(args.get("top_k", "20")), Path("reports/error_gallery.md"),
        Path("reports/figures"), Path("reports/error_gallery_notes.md"),
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    print("wrote reports/error_gallery.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
