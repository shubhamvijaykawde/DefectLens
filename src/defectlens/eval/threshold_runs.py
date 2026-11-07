"""Orchestration of Phase 4a: select thresholds, freeze them, evaluate once, render the report.

The model code is injected (``VisaProvider`` / ``DetectorProvider``) so the protocol can be tested
with fake scores: the selection step calls the providers with the selection split only, the
evaluation step with the evaluation split only.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from defectlens.eval.anomaly import auroc
from defectlens.eval.detection import Detections, GroundTruth
from defectlens.eval.thresholds import (
    EVALUATION_SPLITS,
    SELECTION_SPLITS,
    FrozenEntry,
    ProtocolError,
    entries_binary,
    entries_detection,
    evaluate_entry,
)


@dataclass
class VisaScores:
    """Image scores of one split of one category."""

    paths: list[str]
    scores: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int64]


@dataclass
class DetectorOutput:
    """Detections of one split; ``metrics`` are the shared detection metrics on that split."""

    dets: list[Detections]
    gts: list[GroundTruth]
    names: list[str]
    metrics: dict[str, float]


# (model, category, split, purpose) -> scores ; (run_name, split, purpose) -> detections
VisaProvider = Callable[[str, str, str, str], VisaScores]
DetectorProvider = Callable[[str, str, str], DetectorOutput]


def assert_reproduces(
    name: str, got: npt.ArrayLike, saved: npt.ArrayLike, *, rtol: float = 1e-3, atol: float = 1e-4
) -> float:
    """Raise unless a reloaded model reproduces the values saved at training time.

    Returns the largest absolute difference. A mismatch means the checkpoint or the way it is
    restored is wrong, which must be fixed before any threshold is trusted.
    """
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(saved, dtype=np.float64)
    if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = float(np.abs(a - b).max()) if a.shape == b.shape else float("nan")
        msg = f"{name}: reloaded model does not reproduce the saved values (max |diff| {diff:.3g})"
        raise ProtocolError(msg)
    return float(np.abs(a - b).max()) if a.size else 0.0


def select_all(
    visa: VisaProvider, detect: DetectorProvider, *, models: Sequence[str],
    categories: Sequence[str], detectors: Sequence[str], ratios: Sequence[float], iou: float,
) -> list[FrozenEntry]:
    """Selection step: every (model, category) on ``threshold_calibration``, every detector on
    ``val``. Reads no other split."""
    entries: list[FrozenEntry] = []
    for model in models:
        for category in categories:
            split = SELECTION_SPLITS["visa"]
            s = visa(model, category, split, "selection")
            entries += entries_binary(model, category, split, s.scores, s.labels, ratios)
    for run in detectors:
        split = SELECTION_SPLITS["neu_det"]
        d = detect(run, split, "selection")
        entries += entries_detection(run, split, d.dets, d.gts, ratios, iou)
    return entries


def evaluate_all(
    entries: Sequence[FrozenEntry], visa: VisaProvider, detect: DetectorProvider, iou: float
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Evaluation step: score each (model, category) once on its evaluation split and apply every
    frozen threshold. Returns result rows and a threshold-free context metric per model/category
    (image AUROC, or the detection metrics)."""
    by_key: dict[tuple[str, str, str], list[FrozenEntry]] = defaultdict(list)
    for e in entries:
        by_key[e.key].append(e)
    rows: list[dict[str, Any]] = []
    context: dict[str, dict[str, float]] = {}
    for (domain, model, category), group in by_key.items():
        split = EVALUATION_SPLITS[domain]
        if domain == "visa":
            s = visa(model, category, split, "evaluation")
            context[f"{model}/{category}"] = {"image_auroc": auroc(s.scores, s.labels)}
            rows += [evaluate_entry(e, split, scores=s.scores, labels=s.labels) for e in group]
        else:
            d = detect(model, split, "evaluation")
            context[f"{model}/{category}"] = {
                k: d.metrics[k] for k in ("map50", "map50_95", "recall", "precision")
            }
            rows += [evaluate_entry(e, split, dets=d.dets, gts=d.gts, iou_thr=iou) for e in group]
    return rows, context


def check_not_a_second_evaluation(results_path: Path, frozen_sha256: str) -> None:
    """Refuse a new evaluation if a different frozen file was already evaluated.

    Re-running the same frozen file is a deterministic repeat. Evaluating changed thresholds on
    the same evaluation data is a protocol breach: it has to be disclosed, so it needs a person
    to move the old results file away and write the disclosure.
    """
    if not results_path.is_file():
        return
    previous = json.loads(results_path.read_text()).get("frozen_file_sha256")
    if previous != frozen_sha256:
        msg = (
            f"{results_path} holds an evaluation of a different frozen file ({previous}). "
            "Evaluating changed thresholds on the evaluation split again is a protocol breach: "
            "disclose it in reports/thresholds.md, then move the old file away to proceed"
        )
        raise ProtocolError(msg)


def clean(value: Any) -> Any:
    """JSON-safe copy: NaN/inf become None, tuples become lists, numpy scalars become Python."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# --------------------------------------------------------------------------------------------
# Report


def _f(x: float | None, digits: int = 3) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _ci(pair: Sequence[float] | None, value: float) -> str:
    if pair is None:
        return _f(value)
    return f"{_f(value, 2)} [{_f(pair[0], 2)}, {_f(pair[1], 2)}]"


def _ratio(r: float) -> str:
    return f"{r:g}"


def summarize(entries: Sequence[FrozenEntry], rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Counts that say how well the frozen thresholds carried over to the evaluation split."""
    sel = {(e.model, e.category, e.ratio): e for e in entries}
    visa = [r for r in rows if r["domain"] == "visa"]
    neu = [r for r in rows if r["domain"] == "neu_det"]
    zero_fn = [r for r in visa if sel[(r["model"], r["category"], r["ratio"])].fn == 0]
    return {
        "visa_entries": len(visa),
        "visa_beat_trivial": sum(r["cost"] < r["trivial_cost"] for r in visa),
        "visa_tie_trivial": sum(r["cost"] == r["trivial_cost"] for r in visa),
        "visa_worse_than_trivial": sum(r["cost"] > r["trivial_cost"] for r in visa),
        "visa_cost_per_image_higher_on_evaluation": sum(
            r["cost_per_item"] > r["selection_cost_per_item"] for r in visa),
        "visa_zero_fn_at_selection": len(zero_fn),
        "visa_zero_fn_at_selection_but_fn_on_evaluation": sum(r["fn"] > 0 for r in zero_fn),
        "neu_entries": len(neu),
        "neu_beat_trivial": sum(r["cost"] < r["trivial_cost"] for r in neu),
        "neu_cost_per_box_higher_on_test": sum(
            r["cost_per_item"] > r["selection_cost_per_item"] for r in neu),
    }


def render_report(
    entries: Sequence[FrozenEntry], rows: Sequence[dict[str, Any]],
    context: dict[str, dict[str, float]], frozen_sha256: str, frozen_meta: dict[str, Any],
) -> str:
    """The thresholds report, generated from the frozen file and the one evaluation."""
    ratios = sorted({e.ratio for e in entries})
    sel = {(e.model, e.category, e.ratio): e for e in entries}
    s = summarize(entries, rows)
    lines = [
        "# Cost-based thresholds (Phase 4a)",
        "",
        "Generated by `scripts/evaluate_thresholds.py` from `reports/frozen_thresholds.json`; "
        "nothing is typed by hand (the commentary lives in README.md and PLAN.md).",
        "",
        f"- Cost = `fn_fp_ratio * FN + FP`, ratios {', '.join(_ratio(r) for r in ratios)}. "
        "The ratios are assumptions about a hypothetical line, not measured costs.",
        f"- Frozen file SHA-256: `{frozen_sha256}` (created {frozen_meta.get('created', '?')}, "
        f"git `{frozen_meta.get('git_commit', '?')}`).",
        "- VisA: one image-level threshold per (model, category), selected on "
        "`threshold_calibration`, evaluated once on `threshold_evaluation`. An image is flagged "
        "when `score >= threshold`; FN = missed anomalous image, FP = flagged normal image.",
        "- NEU-DET: one score threshold per detector (all classes), selected on `val`, evaluated "
        f"on `test`. Boxes match at IoU {frozen_meta.get('iou', 0.5)} (class-aware, greedy); "
        "FN = unmatched ground-truth box, FP = kept box that matched nothing.",
        "- The two VisA halves are stratified halves of one test set, so they have the same number "
        "of images per category and selection and evaluation cost are directly comparable. Their "
        "class balance (about 50% anomalous, 67% for cashew) is the benchmark's, not a production "
        "prevalence: the absolute costs would change with a real defect rate.",
        "- `trivial` is the cost of the best constant policy (flag nothing, or flag everything); a "
        "threshold that does not beat it has no value at that ratio. Intervals are 95% Wilson "
        "intervals for the image-level recall and false-positive rate (50 anomalous images and "
        "25-50 normal ones per category, so they are wide).",
        "- Selection used `val` for NEU-DET, which also picked each detector's checkpoint (and "
        "stopped D-FINE-S early), so the selection-split costs there are optimistic by "
        "construction. The NEU-DET `test` split had already been scored once in Phase 2 at a fixed "
        "score of 0.5 (threshold-free AP and a non-tuned operating point); no threshold was "
        "chosen on it.",
        "- Protocol breaches: none known. A second evaluation on changed thresholds would be "
        "listed here.",
        "",
        "## Summary",
        "",
        f"- VisA, {s['visa_entries']} thresholds: {s['visa_beat_trivial']} beat the trivial "
        f"policy on the evaluation split, {s['visa_tie_trivial']} tie it, "
        f"{s['visa_worse_than_trivial']} are worse. The cost per image was higher on evaluation "
        f"than on selection for {s['visa_cost_per_image_higher_on_evaluation']}.",
        f"- VisA: {s['visa_zero_fn_at_selection']} thresholds missed no anomalous image at "
        f"selection; {s['visa_zero_fn_at_selection_but_fn_on_evaluation']} of them missed at "
        "least one on evaluation.",
        f"- NEU-DET, {s['neu_entries']} thresholds: {s['neu_beat_trivial']} beat the trivial "
        f"policy; the cost per box was higher on test than on val for "
        f"{s['neu_cost_per_box_higher_on_test']}.",
        "",
    ]

    for ratio in ratios:
        lines += [f"## VisA, fn_fp_ratio = {_ratio(ratio)}", ""]
        lines += [
            "| category | model | threshold | sel FN/FP | eval FN/FP | eval recall [95% CI] | "
            "eval FPR [95% CI] | sel cost/img | eval cost/img | eval cost | trivial |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in (x for x in rows if x["domain"] == "visa" and x["ratio"] == ratio):
            e = sel[(r["model"], r["category"], ratio)]
            lines.append(
                f"| {r['category']} | {r['model']} | {r['threshold']:.4g} | {e.fn}/{e.fp} | "
                f"{r['fn']}/{r['fp']} | {_ci(r['recall_ci'], r['recall'])} | "
                f"{_ci(r['false_positive_rate_ci'], r['false_positive_rate'])} | "
                f"{_f(r['selection_cost_per_item'])} | {_f(r['cost_per_item'])} | "
                f"{r['cost']:g} | {r['trivial_cost']:g} |"
            )
        lines.append("")

    lines += ["## VisA, total over the four categories", "",
              "| model | fn_fp_ratio | sel cost | eval cost | trivial (eval) |",
              "|---|---|---|---|---|"]
    totals: dict[tuple[str, float], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in (x for x in rows if x["domain"] == "visa"):
        t = totals[(r["model"], r["ratio"])]
        t[0] += sel[(r["model"], r["category"], r["ratio"])].cost
        t[1] += r["cost"]
        t[2] += r["trivial_cost"]
    ordered = sorted(totals.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    for (model, ratio), (c_sel, c_eval, c_triv) in ordered:
        lines.append(f"| {model} | {_ratio(ratio)} | {c_sel:g} | {c_eval:g} | {c_triv:g} |")
    lines.append("")

    lines += ["## VisA, threshold-free context (image AUROC on `threshold_evaluation`)", "",
              "| model | category | image AUROC |", "|---|---|---|"]
    for key, ctx in context.items():
        if "image_auroc" in ctx:
            model, category = key.split("/")
            lines.append(f"| {model} | {category} | {_f(ctx['image_auroc'], 4)} |")
    lines.append("")

    for ratio in ratios:
        lines += [f"## NEU-DET, fn_fp_ratio = {_ratio(ratio)}", "",
                  "| detector | threshold | sel TP/FP/FN | eval TP/FP/FN | eval recall | "
                  "eval precision | sel cost/box | eval cost/box | eval cost | trivial |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in (x for x in rows if x["domain"] == "neu_det" and x["ratio"] == ratio):
            e = sel[(r["model"], r["category"], ratio)]
            lines.append(
                f"| {r['model']} | {r['threshold']:.4g} | {e.tp}/{e.fp}/{e.fn} | "
                f"{r['tp']}/{r['fp']}/{r['fn']} | {_f(r['recall'])} | {_f(r['precision'])} | "
                f"{_f(r['selection_cost_per_item'])} | {_f(r['cost_per_item'])} | "
                f"{r['cost']:g} | {r['trivial_cost']:g} |"
            )
        lines.append("")
    lines += ["## NEU-DET, threshold-free context (`test`, from the reloaded checkpoints)", "",
              "| detector | AP@50 | AP@50:95 | recall@0.5 | precision@0.5 |",
              "|---|---|---|---|---|"]
    for key, ctx in context.items():
        if "map50" in ctx:
            lines.append(
                f"| {key.split('/')[0]} | {_f(ctx['map50'], 4)} | {_f(ctx['map50_95'], 4)} | "
                f"{_f(ctx['recall'], 4)} | {_f(ctx['precision'], 4)} |"
            )
    lines.append("")
    return "\n".join(lines)


def with_notes(report: str, notes: Path | None) -> str:
    """Append the hand-written notes (a file next to the report) or a TODO placeholder."""
    body = notes.read_text().rstrip() if notes is not None and notes.is_file() else (
        "TODO: written after reading the tables; kept in a separate notes file so regenerating "
        "the report does not overwrite them."
    )
    return report.rstrip() + "\n\n## Notes\n\n" + body + "\n"

