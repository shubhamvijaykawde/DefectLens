"""Phase 5a: compare the ONNX models with PyTorch and write `reports/onnx_verification.json`.

Usage:
  uv run python scripts/verify_onnx.py models=dfine_s            # about 5 min incl. the bootstrap
  uv run python scripts/verify_onnx.py models=efficientad
  uv run python scripts/verify_onnx.py models=faster_rcnn,patchcore,anomaly_dino

Reads `models/onnx/*.onnx` (run `scripts/export_onnx.py` first). D-FINE: raw outputs and
detection metrics on NEU-DET `val` and `test` for PyTorch (strict fp32, cuDNN default, AMP) and
ORT CUDA with TF32 off / on, plus a paired bootstrap of the metric differences. Anomaly models:
image scores on VisA `threshold_calibration`. Faster R-CNN: metrics on `val`. A partial run
replaces only the entries it redoes. Each model is logged as an MLflow run (phase 5a).
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
from omegaconf import OmegaConf

from defectlens.export.config import load_export_config
from defectlens.export.verify_run import verify_anomaly, verify_dfine, verify_faster_rcnn
from defectlens.seed import seed_everything
from defectlens.tracking import git_commit_hash, tracked_run


def flat_metrics(entry: dict[str, Any]) -> dict[str, float]:
    """Numbers worth an MLflow metric: per-variant detection metrics / anomaly agreement."""
    out: dict[str, float] = {}
    for split, block in entry.get("splits", {"": entry}).items():
        for variant, d in block.get("variants", {}).items():
            prefix = f"{split}/{variant}".strip("/")
            for k, val in d.get("metrics", {}).items():
                out[f"{prefix}/{k}"] = float(val)
            for k in ("max_abs", "max_rel", "spearman", "auroc_ref", "auroc_got", "max_map_abs"):
                if k in d:
                    out[f"{prefix}/{k}"] = float(d[k])
            for k, val in d.get("raw", {}).items():
                out[f"{prefix}/raw_{k}"] = float(val)
    return out


def main(argv: list[str]) -> int:
    cfg = load_export_config(argv)
    seed_everything(42)
    onnx_dir = Path(cfg.export.out_dir)
    report = Path(cfg.verify.report)
    results: dict[str, Any] = json.loads(report.read_text()) if report.is_file() else {}
    results["environment"] = {
        "torch": torch.__version__, "onnxruntime": ort.__version__,
        "gpu": torch.cuda.get_device_name(0), "git_commit": git_commit_hash(),
    }
    for name in cfg.export.models:
        t0 = time.perf_counter()
        runs: list[tuple[str, dict[str, Any]]] = []
        try:
            if name == "dfine_s":
                runs.append(("dfine_s", verify_dfine(cfg, onnx_dir / "dfine_s.onnx")))
            elif name == "faster_rcnn":
                runs.append(("faster_rcnn", verify_faster_rcnn(cfg, onnx_dir / "faster_rcnn.onnx")))
            else:
                for category in cfg.export.anomaly[name]:
                    path = onnx_dir / f"{name}_{category}.onnx"
                    runs.append((f"{name}/{category}", verify_anomaly(cfg, name, category, path)))
        except Exception as exc:  # record, then carry on with the next model
            reason = f"{type(exc).__name__}: {exc}"[:500]
            runs.append((name, {"status": "failed", "reason": reason}))
        for key, entry in runs:
            old = results.setdefault("models", {}).get(key, {})
            if "splits" in entry and "splits" in old:  # a partial D-FINE run keeps the other split
                entry["splits"] = {**old["splits"], **entry["splits"]}
            results["models"][key] = entry
            if "status" in entry:
                print(f"{key}: FAILED {entry['reason']}")
                continue
            with tracked_run(
                "defectlens", OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
                seed=42, dvc_file=Path("data/raw/neu_det.dvc"), run_name=f"onnx_verify_{key}",
            ):
                mlflow.set_tags({"phase": "5a", "step": "verify", "model": key})
                mlflow.log_metrics(flat_metrics(entry))
                mlflow.log_metric("peak_ram_mib",
                                  resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            print(f"{key}: done in {time.perf_counter() - t0:.0f}s")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {report}; peak RAM "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
