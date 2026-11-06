"""Fit anomaly models on VisA categories and score `threshold_calibration`.

Usage:
  uv run python scripts/train_anomaly.py model=patchcore category=cashew
  uv run python scripts/train_anomaly.py model=patchcore,anomaly_dino \
      category=cashew,candle,pcb1,macaroni2

`model=` and `category=` accept comma lists (every combination runs, one MLflow run each); any
other argument is a Hydra override, e.g. `model.shots=5` or `trainer.fast_dev_run=true`.
`threshold_evaluation` is never read here.
"""

from __future__ import annotations

import sys

from defectlens.anomaly.runner import parse_list, run_anomaly
from defectlens.config import load_config


def main() -> int:
    args = sys.argv[1:]
    models = parse_list(next((a[6:] for a in args if a.startswith("model=")), "patchcore"))
    categories = parse_list(next((a[9:] for a in args if a.startswith("category=")), "cashew"))
    rest = [a for a in args if not a.startswith(("model=", "category="))]
    for model in models:
        cfg = load_config(["data=visa", f"model={model}", *rest])
        for category in categories:
            run_anomaly(cfg, category)
    return 0


if __name__ == "__main__":
    sys.exit(main())
