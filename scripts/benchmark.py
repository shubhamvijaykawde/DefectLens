"""Phase 5a: latency benchmark of the ONNX models on ORT CUDA, and PyTorch for reference.

Usage:
  uv run python scripts/benchmark.py providers=cuda models=dfine_s          # about 1 min
  uv run python scripts/benchmark.py providers=cuda                         # all models, 3-6 min
  uv run python scripts/benchmark.py providers=cuda,cpu models=efficientad  # adds ORT CPU
  uv run python scripts/benchmark.py benchmark.runs=100                     # quick look

Protocol: batch 1, 50 warm-up and 500 timed calls (`benchmark.warmup` / `benchmark.runs`), p50 /
p95 and throughput, each (model, variant) in its own process so RAM and device memory are its
own. Variants: ORT CUDA with TF32 off and on, and PyTorch fp32 / strict fp32 / fp16 AMP.
Anomaly models are timed on `benchmark.category` (pcb1). Results go to `reports/benchmark.json`
(a partial run replaces only the entries it redoes) and to MLflow (phase 5a); build the table
with `scripts/render_benchmark.py`. Needs the ONNX files from `scripts/export_onnx.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import onnxruntime as ort
import torch
from omegaconf import OmegaConf

from defectlens.export.benchmark import variants
from defectlens.export.benchmark_run import DETECTORS, measure
from defectlens.export.config import load_export_config
from defectlens.tracking import git_commit_hash, tracked_run

CHILD = "--child"


def key(r: dict[str, Any]) -> tuple[str, str | None, str]:
    return r["model"], r.get("category"), r["variant"]


def child(argv: list[str]) -> int:
    """Measure one (model, category, variant); print one ``RESULT {json}`` line."""
    model, category, label, *overrides = argv
    cfg = load_export_config(overrides)
    row = measure(cfg, model, None if category == "-" else category, label)
    print("RESULT " + json.dumps(row))
    return 0


def main(argv: list[str]) -> int:
    cfg = load_export_config(argv)
    b = cfg.benchmark
    providers = [p.strip() for p in str(b.providers).replace("[", "").replace("]", "").split(",")]
    plan = variants(providers, [bool(t) for t in b.tf32], list(b.torch_variants))
    report = Path(b.report)
    old: list[dict[str, Any]] = (
        json.loads(report.read_text())["rows"] if report.is_file() else []
    )
    rows: list[dict[str, Any]] = []
    for model in cfg.export.models:
        category = None if model in DETECTORS else str(b.category)
        onnx = Path(cfg.export.out_dir) / (
            f"{model}.onnx" if category is None else f"{model}_{category}.onnx"
        )
        if not onnx.is_file():
            print(f"{model}: no ONNX file {onnx}, skipped (run scripts/export_onnx.py)")
            continue
        for v in plan:
            cmd = [sys.executable, __file__, CHILD, model, category or "-", v.label, *argv]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
            if line is None:
                tail = (proc.stderr.strip().splitlines() or ["no output"])[-1][:300]
                row = {"model": model, "category": category, "variant": v.label,
                       "status": "failed", "reason": tail}
                print(f"{model:13s} {v.label:22s} FAILED {tail}")
            else:
                row = json.loads(line.removeprefix("RESULT "))
                lat = row["latency"]
                print(f"{model:13s} {v.label:22s} p50 {lat['p50_ms']:7.2f} ms  "
                      f"p95 {lat['p95_ms']:7.2f} ms  {lat['throughput_per_s']:7.1f}/s  "
                      f"VRAM {row['device_used_by_variant_mib'] or 0:6.0f} MiB  "
                      f"RAM {row['peak_ram_mib']:5.0f} MiB")
            rows.append(row)

    redone = {key(r) for r in rows}
    merged = [r for r in old if key(r) not in redone] + rows
    env = {"torch": torch.__version__, "onnxruntime": ort.__version__,
           "gpu": torch.cuda.get_device_name(0), "git_commit": git_commit_hash(),
           "platform": "WSL2 Ubuntu", "protocol": {"batch": int(b.batch), "warmup": int(b.warmup),
                                                   "runs": int(b.runs)}}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"environment": env, "rows": merged}, indent=1) + "\n")
    with tracked_run(
        "defectlens", OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        seed=42, dvc_file=Path("data/raw/neu_det.dvc"), run_name="onnx_benchmark",
    ):
        mlflow.set_tags({"phase": "5a", "step": "benchmark"})
        for r in rows:
            if "latency" in r:
                tag = f"{r['model']}_{r.get('category') or 'neu'}/{r['variant']}"
                mlflow.log_metrics({f"{tag}/p50_ms": r["latency"]["p50_ms"],
                                    f"{tag}/p95_ms": r["latency"]["p95_ms"],
                                    f"{tag}/peak_ram_mib": r["peak_ram_mib"]})
        mlflow.log_artifact(str(report))
    print(f"wrote {report} ({len(rows)} new rows)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == CHILD:
        sys.exit(child(sys.argv[2:]))
    sys.exit(main(sys.argv[1:]))
