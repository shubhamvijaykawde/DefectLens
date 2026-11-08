"""Leave-one-defect-type-out on VisA (Phase 4b): the protocol, the numbers and the report.

Question: when a defect type was never in the supervised detector's training data, does the
anomaly path (trained on normal images only) still catch it, and does the detector miss it?

Protocol, per held-out type T (this module is pure: no model, no GPU, no file I/O except the
report text; ``eval/unseen_run.py`` runs the models):

* The detector trains on ``threshold_calibration`` anomalies that carry exactly one defect type,
  never T, plus normal images from ``train``. Images with several types are left out of
  training (the mask cannot be split by type). It has no class for T.
* Every model, detector and anomaly alike, gets one operating point per ``max_fpr``: the lowest
  threshold whose false-alarm rate on the ``threshold_calibration`` *normal* images is at most
  ``max_fpr``. Only normal scores are used, so the detector having trained on calibration
  anomalies does not leak into the threshold, and the models are compared at the same
  false-alarm budget, not at the same score.
* ``threshold_evaluation`` is then scored once per model: recall on its anomalies that contain
  T ("unseen") and on those that do not ("seen", the detector's sanity check), and the realised
  false-alarm rate on its normals.
* Threshold-free image AUROC for T uses every anomalous image that contains T (both halves, none
  of them in any training set) against all test normals.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from defectlens.data.visa_to_coco import Types
from defectlens.eval.anomaly import auroc
from defectlens.eval.thresholds import wilson_interval

CALIBRATION = "threshold_calibration"
EVALUATION = "threshold_evaluation"
NO_HOLDOUT = "none"
DETECTOR = "faster_rcnn"
MODEL_ORDER = (DETECTOR, "patchcore", "anomaly_dino", "efficientad")

Scores = Mapping[str, float]  # image path -> image-level score (higher = more anomalous)


@dataclass(frozen=True)
class TypeSplit:
    """Which images play which role when ``holdout`` is left out. All values are image paths."""

    category: str
    holdout: str | None
    classes: tuple[str, ...]  # detector classes: every defect type except the held-out one
    train_anomalies: tuple[str, ...]  # detector training: calibration, one type, not the holdout
    cal_normals: tuple[str, ...]  # fix every operating point
    eval_normals: tuple[str, ...]  # realised false-alarm rate
    unseen_eval: tuple[str, ...]  # evaluation anomalies that contain the holdout
    seen_eval: tuple[str, ...]  # evaluation anomalies that do not
    unseen_all: tuple[str, ...]  # anomalies containing the holdout in both halves (AUROC)


def make_type_split(
    category: str,
    cal_paths: Sequence[str],
    eval_paths: Sequence[str],
    types: Mapping[str, Types],
    holdout: str | None,
) -> TypeSplit:
    """Roles of the two test halves for one held-out type (``None``: the no-holdout control)."""
    if set(cal_paths) & set(eval_paths):
        msg = "calibration and evaluation halves overlap"
        raise ValueError(msg)
    missing = [p for p in (*cal_paths, *eval_paths) if p not in types]
    if missing:
        msg = f"{len(missing)} images have no type label, e.g. {missing[0]}"
        raise ValueError(msg)
    all_types = sorted({t for p in (*cal_paths, *eval_paths) for t in types[p]})
    if holdout is not None and holdout not in all_types:
        msg = f"holdout {holdout!r} is not a defect type of {category}: {all_types}"
        raise ValueError(msg)

    def has(p: str) -> bool:
        return holdout is not None and holdout in types[p]

    cal, ev = sorted(cal_paths), sorted(eval_paths)
    return TypeSplit(
        category=category,
        holdout=holdout,
        classes=tuple(t for t in all_types if t != holdout),
        train_anomalies=tuple(p for p in cal if len(types[p]) == 1 and not has(p)),
        cal_normals=tuple(p for p in cal if not types[p]),
        eval_normals=tuple(p for p in ev if not types[p]),
        unseen_eval=tuple(p for p in ev if types[p] and has(p)),
        seen_eval=tuple(p for p in ev if types[p] and not has(p)),
        unseen_all=tuple(p for p in (*cal, *ev) if types[p] and has(p)),
    )


def type_counts(types: Mapping[str, Types], paths: Sequence[str]) -> dict[str, int]:
    """Images per defect type among ``paths``; a multi-type image counts under each of its types."""
    out: dict[str, int] = {}
    for p in paths:
        for t in types[p]:
            out[t] = out.get(t, 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------------------------
# Operating points and numbers


def threshold_at_fpr(normal_scores: npt.ArrayLike, max_fpr: float) -> float:
    """Lowest ``t`` such that flagging ``score >= t`` flags at most ``max_fpr`` of the normals.

    The candidates are the distinct normal scores and a value just above the largest one (flags
    no normal image), so the answer always exists and is the most sensitive allowed threshold.
    """
    s = np.asarray(normal_scores, dtype=np.float64).ravel()
    if len(s) == 0:
        msg = "no normal scores to set a threshold from"
        raise ValueError(msg)
    if not 0.0 <= max_fpr <= 1.0:
        msg = f"max_fpr must be in [0, 1], got {max_fpr}"
        raise ValueError(msg)
    candidates = np.append(np.unique(s), np.nextafter(s.max(), np.inf))
    for t in candidates:  # ascending: the first allowed one is the lowest
        if float((s >= t).mean()) <= max_fpr:
            return float(t)
    msg = "unreachable: the last candidate flags nothing"
    raise AssertionError(msg)


def choose_thresholds(
    split: TypeSplit, scores: Mapping[str, Scores], max_fprs: Sequence[float]
) -> dict[str, dict[float, float]]:
    """``{model: {max_fpr: threshold}}`` from the calibration normals' scores and nothing else."""
    return {
        model: {
            a: threshold_at_fpr([sc[p] for p in split.cal_normals], a) for a in max_fprs
        }
        for model, sc in scores.items()
    }


def evaluate_thresholds(
    split: TypeSplit, scores: Mapping[str, Scores], thresholds: Mapping[str, Mapping[float, float]]
) -> list[dict[str, Any]]:
    """One row per (model, max_fpr): counts at the fixed threshold on the evaluation half."""
    rows: list[dict[str, Any]] = []
    for model, sc in scores.items():
        def arr(paths: Sequence[str], sc: Scores = sc) -> npt.NDArray[np.float64]:
            return np.array([sc[p] for p in paths], dtype=np.float64)

        cal_n, eval_n = arr(split.cal_normals), arr(split.eval_normals)
        unseen, seen = arr(split.unseen_eval), arr(split.seen_eval)
        unseen_all = arr(split.unseen_all)
        normals = np.concatenate([cal_n, eval_n])
        auc_unseen = auroc(
            np.concatenate([unseen_all, normals]),
            np.r_[np.ones(len(unseen_all)), np.zeros(len(normals))],
        )
        auc_seen = auroc(
            np.concatenate([seen, normals]), np.r_[np.ones(len(seen)), np.zeros(len(normals))]
        )
        for max_fpr, t in thresholds[model].items():
            rows.append({
                "model": model, "holdout": split.holdout or NO_HOLDOUT, "max_fpr": max_fpr,
                "threshold": t,
                "cal_fp": int((cal_n >= t).sum()), "n_cal_normals": len(cal_n),
                "eval_fp": int((eval_n >= t).sum()), "n_eval_normals": len(eval_n),
                "unseen_tp": int((unseen >= t).sum()), "n_unseen": len(unseen),
                "seen_tp": int((seen >= t).sum()), "n_seen": len(seen),
                "auroc_unseen": auc_unseen, "n_unseen_all": len(unseen_all),
                "auroc_seen": auc_seen,
            })
    return rows


# --------------------------------------------------------------------------------------------
# Report

# Config keys that may differ between runs of the same experiment.
_PER_RUN_KEYS = {"holdout", "smoke", "force"}


def recipe(result: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a run's config that must be identical across held-out types."""
    cfg = result["config"]
    return {
        "unseen": {k: v for k, v in cfg["unseen"].items() if k not in _PER_RUN_KEYS},
        "model": cfg["model"], "preprocess": cfg["preprocess"], "seed": cfg["seed"],
        "trainer": cfg["trainer"],
    }


def check_same_recipe(results: Sequence[Mapping[str, Any]]) -> None:
    """Raise if the results were not all made with one recipe. Changed settings between held-out
    types would make the rows incomparable, and is what tuning on the evaluation half looks like."""
    if results and any(recipe(r) != recipe(results[0]) for r in results[1:]):
        msg = "the result files were made with different recipes; re-run or remove the odd ones"
        raise ValueError(msg)


def _rate(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} = {k / n:.2f} [{lo:.2f}, {hi:.2f}]"


def _num(x: float | None, digits: int = 3) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _find(
    rows: Sequence[Mapping[str, Any]], model: str, max_fpr: float
) -> Mapping[str, Any] | None:
    return next((r for r in rows if r["model"] == model and r["max_fpr"] == max_fpr), None)


def _pct(x: float) -> str:
    return f"{x * 100:g}%"


def render_report(results: Sequence[Mapping[str, Any]]) -> str:
    """The report from the saved per-holdout results (``results`` may be empty: a stub)."""
    runs = {str(r["holdout"]): r for r in results}
    held = [h for h in sorted(runs) if h != NO_HOLDOUT]
    order = [*held, NO_HOLDOUT] if NO_HOLDOUT in runs else held
    max_fprs = sorted({row["max_fpr"] for r in results for row in r["rows"]})
    present = {row["model"] for r in results for row in r["rows"]}
    models = [m for m in MODEL_ORDER if m in present]
    anomaly_models = [m for m in models if m != DETECTOR]
    first = results[0] if results else None
    lines = [
        "# Unseen defect types on VisA pcb1 (Phase 4b)",
        "",
        "Generated by `scripts/render_unseen.py` from `reports/unseen/pcb1_*.json` (one file per "
        "run of `scripts/run_unseen.py`); nothing is typed by hand (the commentary lives in the "
        "notes file and README.md). **Small sample:** see the image counts per defect type below; "
        "every interval is wide.",
        "",
        "## Protocol",
        "",
        "- One run per held-out defect type T, plus a control run that holds nothing out. The "
        "detector is Faster R-CNN ResNet50-FPN v2 (COCO-pretrained), class-aware, with a class "
        "for every defect type except T. It trains on `threshold_calibration` anomalies with "
        "exactly one defect type (never T; the mask cannot be split by type for an image with "
        "several) and a fixed random sample of `train` normal images as negatives. Boxes come "
        "from the VisA masks (`data/visa_to_coco.py`). The recipe is fixed in "
        "`configs/unseen/pcb1.yaml` before any run: no validation set, no checkpoint "
        "selection, the last epoch is used.",
        "- The anomaly models are the Phase 3a/3b checkpoints (trained on `train` normals only, "
        "no anomaly seen), the same for every T.",
        "- Every model gets one operating point per false-alarm budget: the lowest threshold "
        "whose false-alarm rate on the `threshold_calibration` **normal** images is at most "
        "the budget. Scores of anomalous images never enter a threshold. An image is flagged "
        "when `score >= threshold`; a detector's image score is its highest box score. These "
        "are not the cost-based thresholds of Phase 4a (those are frozen and untouched).",
        "- Recall is image-level on `threshold_evaluation`: *unseen* = anomalous images that "
        "contain T, *seen* = those that do not. An image counts as caught if it is flagged, "
        "wherever the evidence is (for the detector: any box, not necessarily on the defect), "
        "which is generous to both kinds of model. Intervals are 95% Wilson intervals.",
        "- `threshold_evaluation` was already used once, for the frozen Phase 4a thresholds, and "
        "is read again here for a different question with thresholds set on calibration normals "
        "only; no Phase 4a threshold or result changes. **Decision (project owner):** this second "
        "read was approved on two conditions: these five runs are the only runs, and the recipe "
        "is not changed after seeing the results. Both held: one MLflow run per holdout, all at "
        "the same commit, with `configs/unseen/pcb1.yaml` untouched since. Any further run or "
        "recipe change would need a new, disclosed decision.",
        "",
    ]

    if first is None:
        lines += ["TODO: no run has been made yet (`scripts/run_unseen.py`); the tables below "
                  "are filled by `scripts/render_unseen.py`.", ""]
        return "\n".join(lines)

    counts = first["type_counts"]
    lines += ["## Images per defect type", "",
              "Images that contain the type (a multi-type image counts under each of its types).",
              "",
              "| type | calibration half | evaluation half | both |", "|---|---|---|---|"]
    for t in sorted(counts["both"]):
        lines.append(
            f"| {t} | {counts['calibration'].get(t, 0)} | {counts['evaluation'].get(t, 0)} | "
            f"{counts['both'][t]} |"
        )
    smallest = min(counts["evaluation"].values(), default=0)
    lines += ["", f"Normal images: {first['n_cal_normals']} in the calibration half, "
              f"{first['n_eval_normals']} in the evaluation half, "
              f"{first['n_train_normals_available']} in `train`. The smallest type has "
              f"{smallest} images in the evaluation half.", ""]

    lines += ["## Detector training", "",
              "| held out | classes | train anomalies | train normals | epochs | iters/epoch | "
              "train s | peak VRAM alloc / reserved (MiB) | final loss | MLflow run |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for h in order:
        d = runs[h]["detector"]
        lines.append(
            f"| {h} | {', '.join(d['classes'])} | {d['n_train_anomalies']} | "
            f"{d['n_train_normals']} | {d['epochs']} | {d['iters_per_epoch']} | "
            f"{_num(d['seconds'], 0)} | {_num(d['peak_vram_alloc_mib'], 0)} / "
            f"{_num(d['peak_vram_reserved_mib'], 0)} | "
            f"{_num(d['loss'][-1] if d['loss'] else None, 4)} | "
            f"`{runs[h].get('mlflow_run_id') or '-'}` |"
        )
    lines.append("")

    head = "| " + " | ".join(["held-out type", "images", *models]) + " |"
    rule = "|" + "---|" * (len(models) + 2)
    for a in max_fprs:
        lines += [f"## Recall on the held-out type, false-alarm budget {_pct(a)} on calibration "
                  "normals", "",
                  "Evaluation half. The detector never saw the type; the anomaly models never "
                  "see any anomaly.", "", head, rule]
        for h in held:
            cells = []
            for m in models:
                row = _find(runs[h]["rows"], m, a)
                cells.append(_rate(row["unseen_tp"], row["n_unseen"]) if row else "-")
            n = next((r["n_unseen"] for r in runs[h]["rows"] if r["max_fpr"] == a), 0)
            lines.append(f"| {h} | {n} | " + " | ".join(cells) + " |")
        lines.append("")

        lines += [f"## Recall on the other (seen) types, budget {_pct(a)}", "",
                  "Evaluation anomalies without the held-out type. Sanity check: if the "
                  "detector is also weak here, a low held-out recall says little about type "
                  "novelty. Rows are the runs; the anomaly models see the same images.", "",
                  "| run (held out) | images | " + " | ".join(models) + " |",
                  "|" + "---|" * (len(models) + 2)]
        for h in order:
            cells = []
            for m in models:
                row = _find(runs[h]["rows"], m, a)
                cells.append(_rate(row["seen_tp"], row["n_seen"]) if row else "-")
            n = next((r["n_seen"] for r in runs[h]["rows"] if r["max_fpr"] == a), 0)
            lines.append(f"| {h} | {n} | " + " | ".join(cells) + " |")
        lines.append("")

        lines += [f"## Operating points and realised false alarms, budget {_pct(a)}", "",
                  "| model | run | threshold | calibration FP | evaluation FP |",
                  "|---|---|---|---|---|"]
        for h in order:
            row = _find(runs[h]["rows"], DETECTOR, a)
            if row:
                lines.append(f"| {DETECTOR} | {h} | {row['threshold']:.4g} | "
                             f"{_rate(row['cal_fp'], row['n_cal_normals'])} | "
                             f"{_rate(row['eval_fp'], row['n_eval_normals'])} |")
        for m in anomaly_models:
            row = _find(first["rows"], m, a)
            if row:
                lines.append(f"| {m} | any | {row['threshold']:.4g} | "
                             f"{_rate(row['cal_fp'], row['n_cal_normals'])} | "
                             f"{_rate(row['eval_fp'], row['n_eval_normals'])} |")
        lines.append("")

    lines += ["## Threshold-free: image AUROC", "",
              "Held-out type: every anomalous image containing T (both halves; none is in any "
              "training set) against all test normals. Seen: evaluation anomalies without T "
              "against the same normals. Same for every budget.", "",
              "| held-out type | images (unseen) | " + " | ".join(models) + " |",
              "|" + "---|" * (len(models) + 2)]
    a0 = max_fprs[0] if max_fprs else 0.0
    for h in held:
        rows = [_find(runs[h]["rows"], m, a0) for m in models]
        n = next((r["n_unseen_all"] for r in runs[h]["rows"]), 0)
        lines.append(f"| {h} | {n} | " + " | ".join(
            _num(r["auroc_unseen"]) if r else "-" for r in rows) + " |")
    lines += ["", "Seen-type AUROC of the same runs:", "",
              "| run (held out) | images (seen) | " + " | ".join(models) + " |",
              "|" + "---|" * (len(models) + 2)]
    for h in order:
        rows = [_find(runs[h]["rows"], m, a0) for m in models]
        n = next((r["n_seen"] for r in runs[h]["rows"]), 0)
        lines.append(f"| {h} | {n} | " + " | ".join(
            _num(r["auroc_seen"]) if r else "-" for r in rows) + " |")
    lines.append("")
    return "\n".join(lines)
