"""Phase 6a: how many selection-split images change side of their frozen threshold under ORT?

Usage:
  uv run python scripts/check_threshold_consistency.py                  # served products + D-FINE
  uv run python scripts/check_threshold_consistency.py products=steel,visa_pcb1

Reads ONLY the selection splits (VisA `threshold_calibration`, NEU-DET `val`; `check_split`
refuses the others). Reference = the Phase 4a PyTorch scoring path, after it has reproduced the
scores / metrics saved at training time. Other side = the serving code on ONNX Runtime with TF32
off and on (`defectlens.serve`). Writes `reports/serve_threshold_consistency.json` and logs an
MLflow run. About 2-3 min on the RTX 4060, under 3 GB RAM, under 2 GB VRAM (one ORT session at
a time).
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import onnxruntime as ort
import torch

from defectlens.eval.scoring import providers
from defectlens.eval.thresholds import FrozenEntry, read_frozen
from defectlens.export.threshold_consistency import anomaly_consistency, detector_consistency
from defectlens.seed import seed_everything
from defectlens.serve.config import ProductSpec, ServeSettings, load_settings
from defectlens.tracking import git_commit_hash, tracked_run

REPORT = Path("reports/serve_threshold_consistency.json")
NEU_IMAGES = Path("data/raw/neu_det/NEU-DET/IMAGES")
# Not served by default, but one config line away: measured so the choice is informed.
CANDIDATES = {
    "candidate_dfine_s": dict(
        format="dfine", onnx=Path("models/onnx/dfine_s.onnx"), frozen_model="dfine_s",
        vram_mib=572,
    ),
}


def entries_of(
    entries: list[FrozenEntry], domain: str, model: str, category: str
) -> list[FrozenEntry]:
    return sorted(
        (e for e in entries if e.key == (domain, model, category)), key=lambda e: e.ratio
    )


def flat(result: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, item in result["items"].items():
        for variant, v in item["result"]["variants"].items():
            for ratio, r in v["by_ratio"].items():
                for k in ("flips", "images_kept_count_differs", "images_alarm_flips"):
                    if k in r:
                        out[f"{key}/{variant}/r{ratio}/{k}"] = float(r[k])
    return out


def main(argv: list[str]) -> int:
    args = dict(a.split("=", 1) for a in argv)
    settings = load_settings(args.get("config"))
    entries, meta = read_frozen(settings.frozen_thresholds)
    iou = float(meta["iou"])
    wanted = args["products"].split(",") if "products" in args else None
    seed_everything(42)

    specs: dict[str, ProductSpec] = dict(settings.products)
    for name, upd in CANDIDATES.items():
        specs[name] = specs["steel"].model_copy(update=upd)
    visa_scores, detect = providers("cuda", verify=True)
    cfg_settings: ServeSettings = settings

    items: dict[str, Any] = {}
    for name, spec in specs.items():
        if wanted is not None and name not in wanted:
            continue
        if not spec.onnx.is_file():
            print(f"{name}: skipped, {spec.onnx} not found")
            continue
        t0 = time.perf_counter()
        domain = "visa" if spec.kind == "anomaly" else "neu_det"
        ents = entries_of(entries, domain, spec.frozen_model, spec.frozen_category)
        if spec.kind == "anomaly":
            ref = visa_scores(spec.frozen_model, spec.frozen_category,
                              "threshold_calibration", "selection")
            result = anomaly_consistency(
                cfg_settings, name, spec, ref.paths, ref.scores, Path("data/raw/visa"), ents
            )
            split = "threshold_calibration"
        else:
            out = detect(spec.frozen_model, "val", "selection")
            result = detector_consistency(
                cfg_settings, name, spec, out.dets, out.gts, out.names, NEU_IMAGES, ents, iou
            )
            split = "val"
        items[name] = {
            "model": spec.frozen_model, "category": spec.frozen_category, "kind": spec.kind,
            "split": split, "onnx": str(spec.onnx), "result": result,
        }
        print(f"{name}: done in {time.perf_counter() - t0:.0f}s")
        for variant, v in result["variants"].items():
            row = {r: x.get("flips", x.get("images_kept_count_differs"))
                   for r, x in v["by_ratio"].items()}
            print(f"    {variant}: {row}")

    report: dict[str, Any] = json.loads(REPORT.read_text()) if REPORT.is_file() else {}
    report["environment"] = {
        "torch": torch.__version__, "onnxruntime": ort.__version__,
        "gpu": torch.cuda.get_device_name(0), "git_commit": git_commit_hash(),
        "frozen_file_content_sha256": json.loads(settings.frozen_thresholds.read_text())[
            "content_sha256"],
    }
    report.setdefault("items", {}).update(items)
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    with tracked_run(
        "defectlens", {"settings": settings.model_dump(mode="json")}, seed=42,
        dvc_file=Path("data/raw/visa.dvc"), run_name="serve_threshold_consistency",
    ):
        mlflow.set_tags({"phase": "6a", "step": "threshold_consistency"})
        mlflow.log_metrics(flat(report))
        mlflow.log_metric("peak_ram_mib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print(f"wrote {REPORT}; peak RAM "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
