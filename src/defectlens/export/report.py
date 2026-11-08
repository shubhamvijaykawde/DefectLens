"""Build ``reports/benchmark.md`` from the export, verification and benchmark result files.

Nothing is typed by hand here: every number comes from ``reports/onnx_export.json``,
``reports/onnx_verification.json`` or ``reports/benchmark.json``. The commentary lives in
``reports/benchmark_notes.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

Json = Mapping[str, Any]

VARIANT_ORDER = ("torch_fp32_strict", "torch_fp32_default", "torch_amp", "ort_tf32_off",
                 "ort_tf32_on")
BENCH_ORDER = ("ort_cuda_tf32_off", "ort_cuda_tf32_on", "ort_cpu", "torch_fp32",
               "torch_fp32_strict", "torch_fp16_amp")


def _f(x: float | None, digits: int = 4) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _e(x: float | None) -> str:
    return "-" if x is None else f"{x:.1e}"


def _ci(t: Sequence[float]) -> str:
    d, lo, hi = t
    return f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def export_section(records: Sequence[Json]) -> list[str]:
    lines = ["## Export", "",
             "Dynamo exporter (`torch.onnx.export(dynamo=True, optimize=False)`), opset 20, "
             "batch 1, CPU, float32. A failed attempt keeps its error text. Files are in "
             "`models/onnx/` (git-ignored).", "",
             "| model | category | exporter | status | size MiB | opset | export s | note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in records:
        opset = ",".join(f"{k}:{v}" for k, v in r.get("opset", {}).items()) or "-"
        note = r["reason"][:200].replace("|", "\\|") if r["status"] != "ok" else ""
        lines.append(
            f"| {r['model']} | {r.get('category') or '-'} | {r['exporter']} | {r['status']} | "
            f"{_f(r.get('size_mib'), 1)} | {opset} | {_f(r.get('seconds'), 1)} | {note} |"
        )
    return [*lines, ""]


def dfine_section(entry: Json) -> list[str]:
    lines = ["## D-FINE-S: ONNX Runtime against PyTorch", "",
             "Reference `torch_fp32_strict` = PyTorch on CUDA with TF32 off in cuDNN and matmul. "
             "`torch_fp32_default` = PyTorch's own default (TF32 convolutions on), "
             "`torch_amp` = the Phase 2 evaluation path (fp16 autocast, here at batch 1). All "
             "variants go through the same decode step (`detection.dfine.decode`), so metric "
             "differences come from the raw network outputs only. Max differences are over every "
             "image, query and class.", ""]
    for split, block in entry.get("splits", {}).items():
        lines += [f"### {split} ({block['n_images']} images, {block['purpose']} split)", "",
                  "| variant | AP@50 | AP@50:95 | recall@0.5 | precision@0.5 | "
                  "images within atol | max abs logits | max abs boxes | sorted-score max abs | "
                  "top-10 overlap | top-100 overlap |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name in VARIANT_ORDER:
            v = block["variants"].get(name)
            if v is None:
                continue
            m, r = v["metrics"], v["raw"]
            lines.append(
                f"| {name} | {_f(m['map50'])} | {_f(m['map50_95'])} | {_f(m['recall'])} | "
                f"{_f(m['precision'])} | {int(r['images_within_atol'])}/{int(r['n_images'])} | "
                f"{_e(r['logits_max_abs'])} | {_e(r['boxes_max_abs'])} | "
                f"{_e(r['sorted_prob_max_abs'])} | {_f(r['top10_overlap'])} | "
                f"{_f(r['top100_overlap'])} |"
            )
        if block.get("bootstrap_95"):
            lines += ["", "Paired image-bootstrap 95% interval of the metric difference (same "
                      "resampled images for both variants):", "",
                      "| difference | AP@50 | AP@50:95 |", "|---|---|---|"]
            for label, vals in block["bootstrap_95"].items():
                lines.append(f"| {label} | {_ci(vals['map50'])} | {_ci(vals['map50_95'])} |")
        lines.append("")
    return lines


def anomaly_section(entries: Mapping[str, Json]) -> list[str]:
    rows = [(k, v) for k, v in entries.items() if "variants" in v and "category" in v]
    if not rows:
        return []
    lines = ["## Anomaly models: ONNX Runtime against PyTorch", "",
             "Image scores on VisA `threshold_calibration` (the selection split), PyTorch strict "
             "fp32 as reference. Relative difference = |d| / max(|ref|, 1e-6).", "",
             "| model | category | images | ORT TF32 | max abs score | max rel score | Spearman | "
             "AUROC PyTorch | AUROC ORT | max abs map |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, e in rows:
        for name, v in e["variants"].items():
            lines.append(
                f"| {e['model']} | {e['category']} | {e['n_images']} | "
                f"{'on' if name.endswith('on') else 'off'} | {_e(v['max_abs'])} | "
                f"{_e(v['max_rel'])} | {_f(v['spearman'], 6)} | {_f(v['auroc_ref'])} | "
                f"{_f(v['auroc_got'])} | {_e(v['max_map_abs'])} |"
            )
    return [*lines, ""]


def faster_rcnn_section(entry: Json) -> list[str]:
    if "variants" not in entry:
        return ["## Faster R-CNN: ONNX Runtime against PyTorch", "",
                f"Not verified: {entry.get('reason', 'no result')}", ""]
    lines = ["## Faster R-CNN: ONNX Runtime against PyTorch (TorchScript export)", "",
             f"NEU-DET `{entry['split']}`, {entry['n_images']} images; torchvision's own "
             "post-processing is inside the graph.", "",
             "| variant | AP@50 | AP@50:95 | recall@0.5 | mean detections per image |",
             "|---|---|---|---|---|"]
    for name, v in entry["variants"].items():
        m = v["metrics"]
        lines.append(f"| {name} | {_f(m['map50'])} | {_f(m['map50_95'])} | {_f(m['recall'])} | "
                     f"{_f(v['mean_detections'], 1)} |")
    return [*lines, ""]


def failures_section(entries: Mapping[str, Json]) -> list[str]:
    bad = [(k, v) for k, v in entries.items() if v.get("status") == "failed"]
    if not bad:
        return []
    return ["## Verification failures", "", "| model | reason |", "|---|---|",
            *[f"| {k} | {v['reason'].replace('|', chr(92) + '|')} |" for k, v in bad], ""]


def latency_section(bench: Json) -> list[str]:
    env, rows = bench["environment"], bench["rows"]
    p = env["protocol"]
    lines = ["## Latency", "",
             f"Batch {p['batch']}, {p['warmup']} warm-up calls, {p['runs']} timed calls of one "
             "fixed selection-split image, one process per row. Model only (no pre- or "
             "post-processing); input from a numpy / CPU tensor, outputs back on the host. "
             "Throughput = 1000 / mean latency, sequential. Device memory = device-wide "
             "`memory.used` peak during the row minus the reading before the process used the "
             "GPU (includes the CUDA context; other GPU users add noise). RAM = process peak "
             "RSS (includes importing torch).", "",
             f"torch {env['torch']}, onnxruntime-gpu {env['onnxruntime']}, {env['gpu']}, "
             f"{env['platform']}, commit `{env['git_commit'][:12]}`.", "",
             "| model | variant | p50 ms | p95 ms | mean ms | throughput /s | device MiB | "
             "RAM MiB |", "|---|---|---|---|---|---|---|---|"]
    order = {v: i for i, v in enumerate(BENCH_ORDER)}
    def sort_key(r: Json) -> tuple[str, str, int]:
        return r["model"], r.get("category") or "", order.get(r["variant"], 9)

    for r in sorted(rows, key=sort_key):
        name = r["model"] + (f" ({r['category']})" if r.get("category") else "")
        if "latency" not in r:
            lines.append(f"| {name} | {r['variant']} | failed: {r['reason'][:120]} | | | | | |")
            continue
        lat = r["latency"]
        lines.append(
            f"| {name} | {r['variant']} | {_f(lat['p50_ms'], 2)} | {_f(lat['p95_ms'], 2)} | "
            f"{_f(lat['mean_ms'], 2)} | {_f(lat['throughput_per_s'], 1)} | "
            f"{_f(r.get('device_used_by_variant_mib'), 0)} | {_f(r['peak_ram_mib'], 0)} |"
        )
    return [*lines, ""]


def render_benchmark(
    export: Sequence[Json] | None, verification: Json | None, bench: Json | None
) -> str:
    lines = ["# ONNX export, verification and benchmark (Phase 5a)", "",
             "Generated by `scripts/render_benchmark.py` from `reports/onnx_export.json`, "
             "`reports/onnx_verification.json` and `reports/benchmark.json`; the commentary is in "
             "`reports/benchmark_notes.md`. **Numbers come from WSL2 on an RTX 4060 Laptop GPU "
             "(8 GB), one machine, one run per row.**", ""]
    lines += (
        export_section(export) if export else ["TODO: `scripts/export_onnx.py` has not run.", ""]
    )
    models = (verification or {}).get("models", {})
    if "dfine_s" in models and "splits" in models["dfine_s"]:
        lines += dfine_section(models["dfine_s"])
    if "faster_rcnn" in models:
        lines += faster_rcnn_section(models["faster_rcnn"])
    lines += anomaly_section(models)
    lines += failures_section(models)
    if not models:
        lines += ["TODO: `scripts/verify_onnx.py` has not run.", ""]
    lines += latency_section(bench) if bench else ["TODO: `scripts/benchmark.py` has not run.", ""]
    return "\n".join(lines)
