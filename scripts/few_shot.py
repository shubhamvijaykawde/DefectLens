"""Few-shot curve: fit AnomalyDINO on k normal images (k = 1, 5, 10) for several saved draws.

Usage:
  uv run python scripts/few_shot.py shots=1,5,10 draws=5
  uv run python scripts/few_shot.py shots=1 draws=1 category=cashew \
      mlflow.tracking_uri=sqlite:///mlruns/few_shot_smoke.db      # smoke, separate MLflow store

`shots=`, `draws=`, `category=` (default: the four VisA categories) and `model=` (default
anomaly_dino) are read here; every other argument is a Hydra override. Draws are read from
`data/splits/few_shot_draws.json` (created on first use, never re-drawn). Runs already finished
in MLflow are skipped, so an interrupted job can simply be started again; `skip_done=false`
repeats them. Scored on `threshold_calibration`; `threshold_evaluation` is never read.
"""

from __future__ import annotations

import sys

from defectlens.anomaly.few_shot import run_few_shot
from defectlens.anomaly.runner import parse_list
from defectlens.config import load_config


def main() -> int:
    args = sys.argv[1:]
    own = ("shots=", "draws=", "category=", "model=", "skip_done=")

    def get(key: str, default: str) -> str:
        return next((a[len(key) :] for a in args if a.startswith(key)), default)

    rest = [a for a in args if not a.startswith(own)]
    categories = parse_list(get("category=", ""))
    if not categories:
        categories = list(load_config(["data=visa"]).data.categories)
    run_few_shot(
        model=get("model=", "anomaly_dino"),
        categories=categories,
        shots=[int(k) for k in parse_list(get("shots=", "1,5,10"))],
        n_draws=int(get("draws=", "5")),
        overrides=rest,
        skip_done=get("skip_done=", "true").lower() != "false",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
