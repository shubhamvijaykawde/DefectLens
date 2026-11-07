"""Phase 4a, step 2: ONE evaluation of the frozen thresholds on the untouched evaluation splits.

Usage:
  uv run python scripts/evaluate_thresholds.py

No tuning options. Needs `reports/frozen_thresholds.json` (refuses to run without it, or if it was
edited after freezing). Reads ONLY `threshold_evaluation` (VisA) and `test` (NEU-DET). If a
different frozen file was already evaluated (`reports/threshold_evaluation.json` exists), it stops:
a second evaluation on changed thresholds is a protocol breach that has to be disclosed first.
Writes `reports/threshold_evaluation.json` and `reports/thresholds.md`, and logs an MLflow run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf

from defectlens.config import load_config
from defectlens.eval.scoring import providers
from defectlens.eval.threshold_runs import (
    check_not_a_second_evaluation,
    clean,
    evaluate_all,
    render_report,
    with_notes,
)
from defectlens.eval.thresholds import file_sha256, read_frozen
from defectlens.tracking import git_commit_hash, tracked_run

FROZEN = Path("reports/frozen_thresholds.json")
RESULTS = Path("reports/threshold_evaluation.json")
REPORT = Path("reports/thresholds.md")
NOTES = Path("reports/thresholds_notes.md")


def main() -> int:
    if len(sys.argv) > 1:
        print("evaluate_thresholds.py has no options (frozen thresholds only)")
        return 2
    entries, meta = read_frozen(FROZEN)  # ProtocolError if missing or edited
    sha = file_sha256(FROZEN)
    check_not_a_second_evaluation(RESULTS, sha)
    cfg = load_config(["data=visa"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} frozen sha256 {sha} ({len(entries)} thresholds)")

    visa, detect = providers(device, verify=False)
    rows, context = evaluate_all(entries, visa, detect, float(meta["iou"]))

    RESULTS.write_text(json.dumps(clean({
        "frozen_file_sha256": sha, "frozen_meta": meta, "git_commit": git_commit_hash(),
        "rows": rows, "context": context,
    }), indent=1) + "\n")
    REPORT.write_text(
        with_notes(render_report(entries, rows, context, sha, meta), NOTES))
    print(f"wrote {RESULTS} and {REPORT}")
    for r in rows:
        print(f"{r['domain']:8s} {r['model']:16s} {r['category']:10s} r={r['ratio']:<3g} "
              f"FN {r['fn']:3d} FP {r['fp']:3d} cost {r['cost']:6g} "
              f"(trivial {r['trivial_cost']:g})")

    config = OmegaConf.to_container(cfg, resolve=True)
    with tracked_run(
        str(cfg.experiment), config, seed=int(cfg.seed),  # type: ignore[arg-type]
        dvc_file=Path(cfg.dvc_file), run_name="thresholds_evaluate",
        tracking_uri=str(cfg.mlflow.tracking_uri), artifact_root=str(cfg.mlflow.artifact_root),
    ):
        mlflow.set_tags({"phase": "4a", "step": "evaluate", "frozen_file_sha256": sha})
        mlflow.log_metrics({
            f"eval/{r['model']}/{r['category']}/r{r['ratio']:g}/{k}": float(r[k])
            for r in rows for k in ("cost", "fn", "fp", "tp")
        })
        for path in (FROZEN, RESULTS, REPORT):
            mlflow.log_artifact(str(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
