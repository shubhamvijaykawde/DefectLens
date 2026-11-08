"""``reports/serve_models.md`` (Phase 6a): what the API serves and how the frozen thresholds fare on
ONNX Runtime. Tables come from ``configs/serve/default.yaml``, ``reports/benchmark.json`` and
``reports/serve_threshold_consistency.json``; the reasoning is in ``reports/serve_models_notes.md``.

The TF32 rule (fixed before the flips were counted): a product runs with TF32 **on** only if, with
TF32 on, none of its frozen thresholds had a flip on the selection split (anomaly: an image on the
other side; detector: an image with a different number of kept boxes) **and** TF32 saves at least
``MIN_SAVING`` of the model's p50 latency in the Phase 5a benchmark. Otherwise it is off.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from defectlens.serve.config import ProductSpec, ServeSettings

Json = Mapping[str, Any]
MIN_SAVING = 0.20
# Which Phase 5a benchmark row stands in for a served model (anomaly models were timed on pcb1).
BENCH_MODEL = {"faster_rcnn_aug": "faster_rcnn", "dfine_s": "dfine_s",
               "efficientad": "efficientad", "patchcore": "patchcore"}


def flip_total(variant: Json) -> int:
    """Flips summed over the frozen ratios of one (product, TF32) result."""
    total = 0
    for r in variant["by_ratio"].values():
        total += int(r["flips"] if "flips" in r else r["images_kept_count_differs"])
    return total


def latency_saving(bench_rows: Sequence[Json], model: str) -> tuple[float, float, float]:
    """``(p50 TF32 off, p50 TF32 on, relative saving)`` of the Phase 5a ORT CUDA rows."""
    p50 = {
        r["variant"]: float(r["latency"]["p50_ms"]) for r in bench_rows
        if r["model"] == BENCH_MODEL[model] and r["variant"].startswith("ort_cuda_tf32")
    }
    off, on = p50["ort_cuda_tf32_off"], p50["ort_cuda_tf32_on"]
    return off, on, (off - on) / off


def tf32_decision(flips_with_tf32_on: int, saving: float, min_saving: float = MIN_SAVING) -> bool:
    return flips_with_tf32_on == 0 and saving >= min_saving


def decide_all(
    settings: ServeSettings, consistency: Json, bench_rows: Sequence[Json]
) -> dict[str, dict[str, Any]]:
    """The rule applied to every served product that has a consistency result."""
    out: dict[str, dict[str, Any]] = {}
    for name, spec in settings.products.items():
        item = consistency["items"].get(name)
        if item is None:
            continue
        flips_on = flip_total(item["result"]["variants"]["tf32_on"])
        off, on, saving = latency_saving(bench_rows, spec.frozen_model)
        out[name] = {
            "flips_tf32_on": flips_on,
            "flips_tf32_off": flip_total(item["result"]["variants"]["tf32_off"]),
            "p50_off_ms": off, "p50_on_ms": on, "saving": saving,
            "tf32": tf32_decision(flips_on, saving), "configured": spec.tf32,
        }
    return out


def _spec_row(name: str, spec: ProductSpec, dec: Mapping[str, Any] | None) -> str:
    saving = "-" if dec is None else f"{dec['saving']:.0%}"
    return (
        f"| {name} | {spec.kind} | {spec.frozen_model} | `{spec.onnx}` | "
        f"{'on' if spec.tf32 else 'off'} | {spec.vram_mib} | {saving} |"
    )


def _onoff(flag: bool) -> str:
    return "on" if flag else "off"


def _visa_rows(consistency: Json) -> list[str]:
    rows = []
    for name, item in consistency["items"].items():
        if item["kind"] != "anomaly":
            continue
        for tf, v in item["result"]["variants"].items():
            br = v["by_ratio"]
            ratios = sorted(br, key=float)
            thresholds = " / ".join(format(br[r]["threshold"], ".4g") for r in ratios)
            flips = " / ".join(str(br[r]["flips"]) for r in ratios)
            first = br[ratios[0]]
            rows.append(
                f"| {name} | {item['model']} | {item['result']['n_images']} | {tf[5:]} | "
                f"{v['score_difference']['max_rel']:.2e} | {thresholds} | {flips} | "
                f"{first['margin_ref']:.2%} / {first['margin_got']:.2%} |"
            )
    return rows


def _detector_rows(consistency: Json) -> list[str]:
    rows = []
    for name, item in consistency["items"].items():
        if item["kind"] != "detector":
            continue
        for tf, v in item["result"]["variants"].items():
            for ratio in sorted(v["by_ratio"], key=float):
                r = v["by_ratio"][ratio]
                c0, c1 = r["ref"]["cost"], r["got"]["cost"]
                differs = f"{r['images_kept_count_differs']} of {r['n_images']}"
                kept = f"{r['kept_boxes_ref']} -> {r['kept_boxes_got']}"
                rows.append(
                    f"| {name} ({item['model']}) | {tf[5:]} | {ratio} | {r['threshold']:.4g} | "
                    f"{differs} | {r['images_alarm_flips']} | {kept} | "
                    f"{c0:g} -> {c1:g} ({(c1 - c0) / c0:+.1%}) |"
                )
    return rows


def render_serve_models(
    settings: ServeSettings, consistency: Json, bench_rows: Sequence[Json]
) -> str:
    env = consistency["environment"]
    dec = decide_all(settings, consistency, bench_rows)
    sha = env["frozen_file_content_sha256"][:16]
    lines = [
        "# What the API serves, and the frozen thresholds on ONNX Runtime (Phase 6a)",
        "",
        "Tables generated by `scripts/render_serve_models.py` from `configs/serve/default.yaml`, "
        "`reports/benchmark.json` and `reports/serve_threshold_consistency.json` (from "
        "`scripts/check_threshold_consistency.py`); the reasoning is the notes at the end.",
        "",
        f"torch {env['torch']}, onnxruntime {env['onnxruntime']}, {env['gpu']}, WSL2, commit "
        f"`{env['git_commit'][:12]}`; frozen file content hash `{sha}...`.",
        "",
        "## Served products",
        "",
        "One model per product. `device MiB` is the declared size of one session (Phase 5a "
        "benchmark, TF32 off; the sum is checked against `vram_budget_mib`). `TF32 saving` is the "
        "p50 model-latency saving of TF32 in that benchmark (anomaly models were timed on pcb1).",
        "",
        "| product | kind | model | ONNX file | TF32 | device MiB | TF32 saving |",
        "|---|---|---|---|---|---|---|",
        *[_spec_row(n, s, dec.get(n)) for n, s in settings.products.items()],
        "",
        f"Budget `vram_budget_mib` = {settings.vram_budget_mib}; default operating point "
        f"`fn_fp_ratio` = {settings.fn_fp_ratio:g}.",
        "",
        "## TF32 rule applied",
        "",
        f"TF32 on only if there were no flips with TF32 on and the saving is at least "
        f"{MIN_SAVING:.0%} (rule fixed before the flips were counted).",
        "",
        "| product | flips TF32 off | flips TF32 on | p50 off ms | p50 on ms | saving | "
        "rule says | configured |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, d in dec.items():
        lines.append(
            f"| {n} | {d['flips_tf32_off']} | {d['flips_tf32_on']} | {d['p50_off_ms']:.2f} | "
            f"{d['p50_on_ms']:.2f} | {d['saving']:.0%} | {_onoff(d['tf32'])} | "
            f"{_onoff(d['configured'])} |"
        )
    lines += [
        "",
        "## Frozen thresholds on ONNX Runtime: VisA products",
        "",
        "Selection split `threshold_calibration` only. Reference = the Phase 4a PyTorch scoring "
        "path (cuDNN default, i.e. TF32 convolutions on), after it reproduced the saved scores. "
        "Other side = the serving code on ORT CUDA. A flip is an image with "
        "`(PyTorch score >= t) != (ORT score >= t)`. `nearest` is the smallest "
        "`|score - t| / t` over the images under each path (ratio 5 threshold).",
        "",
        "| product | model | images | ORT TF32 | max rel score diff | "
        "threshold (ratio 5 / 10 / 50) | flips (5 / 10 / 50) | nearest (PyTorch / ORT) |",
        "|---|---|---|---|---|---|---|---|",
        *_visa_rows(consistency),
        "",
        "## Frozen thresholds on ONNX Runtime: detectors",
        "",
        "NEU-DET `val` (selection split), 270 images. Reference = Phase 4a path (fp16 autocast), "
        "after it reproduced the stored val metrics. NEU-DET has no defect-free images, so an "
        "image-level alarm can not flip in a meaningful way; the counts that matter are the "
        "images whose number of boxes kept at `score >= t` differs and the cost "
        "`ratio * FN + FP` at `t`. `candidate_dfine_s` is not served; it is one config line away.",
        "",
        "| product | ORT TF32 | ratio | threshold | images: different kept boxes | "
        "images: alarm flips | kept boxes PyTorch -> ORT | cost PyTorch -> ORT |",
        "|---|---|---|---|---|---|---|---|",
        *_detector_rows(consistency),
    ]
    return "\n".join(lines) + "\n"
