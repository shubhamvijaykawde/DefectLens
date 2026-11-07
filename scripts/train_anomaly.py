"""Fit anomaly models on VisA categories and score `threshold_calibration`.

Usage:
  uv run python scripts/train_anomaly.py model=patchcore category=cashew
  uv run python scripts/train_anomaly.py model=patchcore,anomaly_dino \
      category=cashew,candle,pcb1,macaroni2
  uv run python scripts/train_anomaly.py model=efficientad category=cashew \
      model.max_steps=500 +run_suffix=_smoke500       # short smoke run, kept apart in MLflow

`model=` and `category=` accept comma lists (every combination runs, one MLflow run each); any
other argument is a Hydra override, e.g. `model.shots=5` or `trainer.fast_dev_run=true`.
`+run_suffix=_x` appends `_x` to the MLflow run name and to the output and checkpoint folders.
`threshold_evaluation` is never read here.
"""

from __future__ import annotations

import subprocess
import sys

from defectlens.anomaly.runner import child_commands, parse_list, run_anomaly
from defectlens.config import load_config


def main() -> int:
    args = sys.argv[1:]
    models = parse_list(next((a[6:] for a in args if a.startswith("model=")), "patchcore"))
    categories = parse_list(next((a[9:] for a in args if a.startswith("category=")), "cashew"))
    rest = [a for a in args if not a.startswith(("model=", "category="))]
    jobs = child_commands(models, categories, rest)
    if len(jobs) > 1:  # one process per run: per-run peak RAM, GPU memory freed in between
        failed = 0
        for job in jobs:
            code = subprocess.run([sys.executable, __file__, *job], check=False).returncode
            if code:
                print(f"FAILED ({code}): {' '.join(job)}", flush=True)
                failed += 1
        return 1 if failed else 0
    cfg = load_config(["data=visa", f"model={models[0]}", *rest])
    run_anomaly(cfg, categories[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
