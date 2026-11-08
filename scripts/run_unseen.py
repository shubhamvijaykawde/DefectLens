"""Phase 4b: leave-one-defect-type-out on VisA pcb1.

Usage:
  uv run python scripts/run_unseen.py category=pcb1 holdout=melt
  uv run python scripts/run_unseen.py category=pcb1 holdout=bent,melt,missing,scratch,none
  uv run python scripts/run_unseen.py holdout=melt unseen.smoke=true      # path check, ~3 min

`holdout` is a defect type, or `none` for the control that holds nothing out. A comma list runs
one child process per value (so peak RAM and VRAM are per run). `category` and `holdout` are
shortcuts for `unseen.category` / `unseen.holdout`; any other `key=value` overrides the merged
config (`configs/unseen/pcb1.yaml` over `data=visa model=faster_rcnn`).

Trains one Faster R-CNN without the held-out type, scores the Phase 3 anomaly checkpoints, fixes
every operating point on the calibration normals, reads `threshold_evaluation` once, and writes
`reports/unseen/<category>_<holdout>.json` plus an MLflow run (phase tag 4b). It refuses to
overwrite an existing result unless `unseen.force=true`. Build the report with
`scripts/render_unseen.py`.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from defectlens.config import load_config
from defectlens.eval.scoring import providers
from defectlens.eval.unseen_run import AnomalyScorer, run_unseen

EXPERIMENT = Path(__file__).resolve().parents[1] / "configs" / "unseen" / "pcb1.yaml"
SHORTCUTS = ("category", "holdout")


def compose(args: list[str]) -> DictConfig:
    """Root config + the experiment file + command-line overrides (later wins)."""
    cfg = load_config(["data=visa", "model=faster_rcnn"])
    OmegaConf.set_struct(cfg, False)  # the experiment file adds keys the root config lacks
    dotlist = [f"unseen.{a}" if a.partition("=")[0] in SHORTCUTS else a for a in args]
    merged = OmegaConf.merge(cfg, OmegaConf.load(EXPERIMENT), OmegaConf.from_dotlist(dotlist))
    assert isinstance(merged, DictConfig)
    return merged


def make_scorer(device: str) -> AnomalyScorer:
    """Reload a Phase 3 checkpoint and score one split. Scoring the calibration half also checks
    that the reloaded model reproduces the scores saved at training time."""
    verified, _ = providers(device, verify=True)
    plain, _ = providers(device, verify=False)

    def scorer(model: str, category: str, split: str, purpose: str) -> Mapping[str, float]:
        s = (verified if purpose == "selection" else plain)(model, category, split, purpose)
        return dict(zip(s.paths, (float(x) for x in s.scores), strict=True))

    return scorer


def main() -> int:
    args = sys.argv[1:]
    holdout = next((a.partition("=")[2] for a in args if a.startswith("holdout=")), "")
    if "," in holdout:  # one process per holdout
        rest = [a for a in args if not a.startswith("holdout=")]
        for h in holdout.split(","):
            cmd = [sys.executable, str(Path(__file__).resolve()), f"holdout={h}", *rest]
            print(f"\n=== {' '.join(cmd[1:])}", flush=True)
            code = subprocess.run(cmd, check=False).returncode
            if code:
                return code
        return 0
    cfg = compose(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} holdout={cfg.unseen.holdout} smoke={cfg.unseen.smoke}")
    run_unseen(cfg, make_scorer(device), device=torch.device(device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
