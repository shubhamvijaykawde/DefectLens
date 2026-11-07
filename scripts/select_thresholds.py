"""Phase 4a, step 1: choose cost-based thresholds and freeze them.

Usage:
  uv run python scripts/select_thresholds.py cost.fn_fp_ratio=5,10,50
  uv run python scripts/select_thresholds.py out=/tmp/frozen_test.json      # another output path

Reads ONLY `threshold_calibration` (VisA: patchcore, anomaly_dino, efficientad x 4 categories) and
`val` (NEU-DET: faster_rcnn, faster_rcnn_aug, dfine_s); `check_split` refuses any other split.
Every reloaded model must first reproduce the scores or metrics saved at training time. Writes
`reports/frozen_thresholds.json` (with an embedded content hash) and logs an MLflow run.
`cost.fn_fp_ratio` takes a comma list; other arguments are Hydra overrides.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf

from defectlens.config import load_config
from defectlens.eval.scoring import DETECTOR_CHECKPOINTS, providers
from defectlens.eval.threshold_runs import select_all
from defectlens.eval.thresholds import SELECTION_SPLITS, file_sha256, write_frozen
from defectlens.tracking import dataset_version, git_commit_hash, tracked_run

MODELS = ("patchcore", "anomaly_dino", "efficientad")


def main() -> int:
    args = sys.argv[1:]

    def get(key: str) -> str | None:
        return next((a[len(key) :] for a in args if a.startswith(key)), None)

    ratios_arg = get("cost.fn_fp_ratio=")
    out = Path(get("out=") or "reports/frozen_thresholds.json")
    rest = [a for a in args if not a.startswith(("cost.fn_fp_ratio=", "out="))]
    cfg = load_config(["data=visa", *rest])
    ratios = (
        [float(r) for r in ratios_arg.split(",") if r] if ratios_arg
        else [float(r) for r in cfg.cost.fn_fp_ratio]
    )
    iou = float(cfg.cost.iou)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ratios={ratios} iou={iou}")

    visa, detect = providers(device, verify=True)
    entries = select_all(
        visa, detect, models=MODELS, categories=list(cfg.data.categories),
        detectors=list(DETECTOR_CHECKPOINTS), ratios=ratios, iou=iou,
    )
    meta = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit_hash(),
        "cost_model": "cost = fn_fp_ratio * FN + FP",
        "ratios": ratios, "iou": iou, "selection_splits": SELECTION_SPLITS,
        "visa_dataset": dataset_version(Path(cfg.dvc_file)),
    }
    sha = write_frozen(out, entries, meta)
    print(f"\nfrozen {len(entries)} thresholds -> {out} (sha256 {sha})\n")
    print("domain   model            category   ratio  threshold   sel FN  FP   cost")
    for e in entries:
        print(f"{e.domain:8s} {e.model:16s} {e.category:10s} {e.ratio:5g}  {e.threshold:10.5g}  "
              f"{e.fn:5d} {e.fp:4d} {e.cost:6g}")

    config = OmegaConf.to_container(cfg, resolve=True)
    with tracked_run(
        str(cfg.experiment), config, seed=int(cfg.seed),  # type: ignore[arg-type]
        dvc_file=Path(cfg.dvc_file), run_name="thresholds_select",
        tracking_uri=str(cfg.mlflow.tracking_uri), artifact_root=str(cfg.mlflow.artifact_root),
    ):
        mlflow.set_tags({"phase": "4a", "step": "select", "frozen_file_sha256": sha})
        mlflow.log_params({"ratios": str(ratios), "iou": iou, "n_entries": len(entries)})
        mlflow.log_metrics({
            f"select/{e.model}/{e.category}/r{e.ratio:g}/{k}": float(v)
            for e in entries for k, v in (("cost", e.cost), ("threshold", e.threshold),
                                          ("fn", e.fn), ("fp", e.fp))
        })
        mlflow.log_artifact(str(out))
    print(f"frozen file sha256: {file_sha256(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
