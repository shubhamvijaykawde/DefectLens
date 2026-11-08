"""Phase 5a: export the trained models to ONNX and record every attempt.

Usage:
  uv run python scripts/export_onnx.py                      # all five models
  uv run python scripts/export_onnx.py models=dfine_s,efficientad

Writes `models/onnx/*.onnx` (git-ignored) and `reports/onnx_export.json` (one record per attempt,
failures included; a partial run only replaces the records it redoes). About 1 minute, CPU only,
under 2 GB of RAM except AnomalyDINO (a 450 MB memory bank, about 3 GB). Verification against
PyTorch is a separate step: `scripts/verify_onnx.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from defectlens.export.config import load_export_config
from defectlens.export.onnx_export import (
    ExportRecord,
    export_anomaly,
    export_dfine,
    export_faster_rcnn,
)


def key(r: dict[str, object]) -> tuple[object, object, object]:
    return r["model"], r.get("category"), r.get("exporter")


def main(argv: list[str]) -> int:
    cfg = load_export_config(argv)
    ex = cfg.export
    out = Path(ex.out_dir)
    records: list[ExportRecord] = []
    for name in ex.models:
        if name == "dfine_s":
            records.append(export_dfine(Path(ex.detectors.dfine_s), out / "dfine_s.onnx"))
        elif name == "faster_rcnn":
            ckpt, path = Path(ex.detectors.faster_rcnn), out / "faster_rcnn.onnx"
            first = export_faster_rcnn(ckpt, path, dynamo=True)
            records.append(first)
            if first.status != "ok":  # Dynamo can not trace torchvision's detection head
                records.append(export_faster_rcnn(ckpt, path, dynamo=False))
        elif name in ex.anomaly:
            for category in ex.anomaly[name]:
                records.append(export_anomaly(name, category, out / f"{name}_{category}.onnx"))
        else:
            msg = f"unknown model {name!r}"
            raise ValueError(msg)

    report = Path(ex.report)
    old = json.loads(report.read_text()) if report.is_file() else []
    new = [r.to_dict() for r in records]
    merged = [o for o in old if key(o) not in {key(n) for n in new}] + new
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(merged, indent=1) + "\n")
    for r in records:
        tail = f"{r.size_mib:.1f} MiB, opset {r.opset}" if r.status == "ok" else r.reason[:140]
        print(f"{r.model:13s} {r.category or '-':10s} {r.exporter:11s} {r.status:6s} "
              f"{r.seconds or 0:5.1f}s  {tail}")
    return 0 if all(r.status == "ok" or r.model == "faster_rcnn" for r in records) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
