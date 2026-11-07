"""Error gallery (Phase 4a, step 3): the worst misses and false alarms at the frozen thresholds.

Reads the evaluation splits (like step 2) and changes no threshold. Colour is never the only cue:
boxes differ by line style as well as colour, and the figures have legends. Anomaly overlays use a
single-hue sequential colour map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from defectlens.eval.anomaly import auroc
from defectlens.eval.detection import Detections, GroundTruth, box_iou, match_greedy
from defectlens.eval.thresholds import FrozenEntry

# Box styles: (colour, line style). Ground truth and predictions differ in style too.
NEAR_IOU = 0.1  # a miss with a same-class kept box above this IoU is a localisation miss
STYLE = {"gt": ("#2a7f3f", "-"), "missed": ("#b8860b", "--"), "tp": ("#1f5fa8", "-"),
         "fp": ("#c0392b", ":")}


# --------------------------------------------------------------------------------------------
# VisA: image level


def rank_visa_cases(
    paths: list[str], scores: npt.ArrayLike, labels: npt.ArrayLike, areas: npt.ArrayLike,
    threshold: float, top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    """The ``top_k`` lowest-scoring anomalous images (``missed`` where below the threshold) and
    the ``top_k`` highest-scoring normal images (``false alarm`` where at or above it)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    a = np.asarray(areas, dtype=np.float64)
    anomalous = np.flatnonzero(y)[np.argsort(s[y], kind="stable")][:top_k]
    normal = np.flatnonzero(~y)[np.argsort(-s[~y], kind="stable")][:top_k]

    def row(i: int, hit: str) -> dict[str, Any]:
        return {"index": int(i), "path": paths[i], "score": float(s[i]),
                "margin": float(s[i] - threshold), "status": hit, "area": float(a[i])}

    return {
        "anomalous": [row(i, "missed" if s[i] < threshold else "caught") for i in anomalous],
        "normal": [row(i, "false alarm" if s[i] >= threshold else "ok") for i in normal],
    }


def area_summary(scores: npt.ArrayLike, labels: npt.ArrayLike, areas: npt.ArrayLike,
                 threshold: float) -> dict[str, float]:
    """Counts of missed / caught anomalous images and their median defect area (fraction)."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    a = np.asarray(areas, dtype=np.float64)
    missed, caught = y & (s < threshold), y & (s >= threshold)
    return {
        "n_anomalous": int(y.sum()), "n_missed": int(missed.sum()), "n_caught": int(caught.sum()),
        "median_area_missed": float(np.median(a[missed])) if missed.any() else float("nan"),
        "median_area_caught": float(np.median(a[caught])) if caught.any() else float("nan"),
        "n_false_alarms": int((~y & (s >= threshold)).sum()), "n_normal": int((~y).sum()),
    }


def best_by_calibration_auroc(models: list[str], category: str, outputs: Path) -> str:
    """The model with the highest image AUROC on ``threshold_calibration`` (saved scores)."""
    best, best_auc = models[0], -1.0
    for m in models:
        saved = json.loads((outputs / f"{m}_{category}" / "calibration_scores.json").read_text())
        auc = auroc(saved["scores"], saved["labels"])
        if auc > best_auc:
            best, best_auc = m, auc
    return best


def plot_visa_cases(
    images: npt.NDArray[np.float32], maps: npt.NDArray[np.float32],
    masks: npt.NDArray[np.bool_], cases: dict[str, list[dict[str, Any]]], threshold: float,
    title: str, out: Path, columns: int = 6,
) -> None:
    """Top row: lowest-scoring anomalous images; bottom row: highest-scoring normal images.

    The anomaly map is overlaid with one single-hue sequential scale shared by the whole figure
    (from the 90th to the 99.9th percentile of all maps shown; weaker values are transparent);
    the ground-truth defect outline is white.
    """
    lo, hi = (float(v) for v in np.percentile(maps, [90, 99.9]))  # transparent below the 90th
    fig, axes = plt.subplots(2, columns, figsize=(2.3 * columns, 5.4), squeeze=False)
    for r, (key, caption) in enumerate((("anomalous", "anomalous, lowest score"),
                                        ("normal", "normal, highest score"))):
        for c in range(columns):
            ax = axes[r][c]
            ax.axis("off")
            if c >= len(cases[key]):
                continue
            case = cases[key][c]
            i = case["index"]
            ax.imshow(images[i])
            ax.imshow(np.ma.masked_less(maps[i], lo), cmap="Purples", vmin=lo, vmax=hi, alpha=0.7)
            if masks[i].any():
                ax.contour(masks[i].astype(float), levels=[0.5], colors="white", linewidths=1.2)
            ax.set_title(f"{case['status']}\nscore {case['score']:.3g} (thr {threshold:.3g})",
                         fontsize=8)
        axes[r][0].text(-0.05, 0.5, caption, transform=axes[r][0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# NEU-DET: box level


@dataclass
class ImageErrors:
    """Box-level outcome for one image at one score threshold."""

    boxes: npt.NDArray[np.float64]  # kept detections
    labels: npt.NDArray[np.int64]
    scores: npt.NDArray[np.float64]
    pred_is_tp: npt.NDArray[np.bool_]
    gt_missed: npt.NDArray[np.bool_]
    gt_near: npt.NDArray[np.bool_]  # missed, but a kept same-class box overlaps it

    @property
    def tp(self) -> int:
        return int(self.pred_is_tp.sum())

    @property
    def fp(self) -> int:
        return int((~self.pred_is_tp).sum())

    @property
    def fn(self) -> int:
        return int(self.gt_missed.sum())


def image_errors(det: Detections, gt: GroundTruth, threshold: float, iou_thr: float = 0.5
                 ) -> ImageErrors:
    """Which kept detections are true positives and which ground-truth boxes are missed
    (class-aware greedy matching, the same rule as every other number in the repo)."""
    keep = det.scores >= threshold
    boxes, scores, labels = det.boxes[keep], det.scores[keep], det.labels[keep]
    tp = np.zeros(len(boxes), dtype=bool)
    hit = np.zeros(len(gt.labels), dtype=bool)
    for c in np.unique(np.concatenate([labels, gt.labels])):
        di = np.flatnonzero(labels == c)
        order = di[np.argsort(-scores[di], kind="stable")]
        gi = np.flatnonzero(gt.labels == c)
        match = match_greedy(box_iou(boxes[order], gt.boxes[gi]), iou_thr)
        for d, g in zip(order, match, strict=True):
            if g >= 0:
                tp[d] = True
                hit[gi[g]] = True
    near = np.zeros(len(gt.labels), dtype=bool)
    for g in np.flatnonzero(~hit):
        same = labels == gt.labels[g]
        if same.any():
            near[g] = box_iou(boxes[same], gt.boxes[g : g + 1]).max() >= NEAR_IOU
    return ImageErrors(boxes, labels, scores, tp, ~hit, near)


def worst_images(errors: list[ImageErrors], names: list[str], ratio: float, top_k: int
                 ) -> list[dict[str, Any]]:
    """Images ranked by ``ratio * FN + FP`` (ties by name), worst first."""
    rows: list[dict[str, Any]] = [
        {"index": i, "name": names[i], "fn": e.fn, "fp": e.fp, "tp": e.tp,
         "cost": ratio * e.fn + e.fp} for i, e in enumerate(errors)
    ]
    rows.sort(key=lambda r: (-float(r["cost"]), str(r["name"])))
    return rows[:top_k]


def class_breakdown(
    errors: list[ImageErrors], gts: list[GroundTruth], class_names: dict[int, str]
) -> list[dict[str, Any]]:
    """Per class: ground-truth boxes, missed ones, and kept boxes of that class matching nothing."""
    out = []
    for c, name in sorted(class_names.items()):
        n_gt = sum(int((g.labels == c).sum()) for g in gts)
        missed = sum(int((e.gt_missed & (g.labels == c)).sum()) for e, g in zip(errors, gts,
                                                                              strict=True))
        near = sum(int((e.gt_near & (g.labels == c)).sum()) for e, g in zip(errors, gts,
                                                                          strict=True))
        fp = sum(int((~e.pred_is_tp & (e.labels == c)).sum()) for e in errors)
        out.append({"class": name, "n_gt": n_gt, "missed": missed, "missed_overlapped": near,
                    "false_alarms": fp})
    return out


def plot_neu_cases(
    images_dir: Path, names: list[str], errors: list[ImageErrors], gts: list[GroundTruth],
    worst: list[dict[str, Any]], class_names: dict[int, str], title: str, out: Path,
    columns: int = 4,
) -> None:
    """The worst images with ground truth (green), matched detections (blue, solid), false
    alarms (red, dotted) and missed ground truth (dashed gold)."""
    rows = int(np.ceil(len(worst) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, w in zip(axes.ravel(), worst, strict=False):
        i = w["index"]
        gray = cv2.imread(str(images_dir / names[i]), cv2.IMREAD_GRAYSCALE)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        e, gt = errors[i], gts[i]
        for box, lab, missed in zip(gt.boxes, gt.labels, e.gt_missed, strict=True):
            kind = "missed" if missed else "gt"
            _rect(ax, box, *STYLE[kind], class_names[int(lab)] if missed else None)
        for box, lab, sc, ok in zip(e.boxes, e.labels, e.scores, e.pred_is_tp, strict=True):
            _rect(ax, box, *STYLE["tp" if ok else "fp"], f"{class_names[int(lab)]} {sc:.2f}")
        ax.set_title(f"{names[i]}: FN {e.fn}, FP {e.fp}", fontsize=8)
    handles = [Line2D([0], [0], color=STYLE[k][0], ls=STYLE[k][1], lw=1.6, label=label)
               for k, label in (("gt", "ground truth (found)"), ("missed", "ground truth (missed)"),
                                ("tp", "detection, correct"), ("fp", "detection, false alarm"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _rect(ax: Any, box: npt.NDArray[np.float64], colour: str, style: str, label: str | None
          ) -> None:
    x1, y1, x2, y2 = (float(v) for v in box)
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=colour, ls=style, lw=1.6))
    if label:
        ax.text(x1, max(y1 - 2, 6), label, color=colour, fontsize=6, backgroundcolor="#ffffffaa")


# --------------------------------------------------------------------------------------------
# Markdown


def _pct(x: float) -> str:
    return "-" if np.isnan(x) else f"{100 * x:.2f}%"


def case_table(cases: list[dict[str, Any]]) -> list[str]:
    lines = ["| # | image | score | score - thr | outcome | defect area |",
             "|---|---|---|---|---|---|"]
    for n, c in enumerate(cases, 1):
        lines.append(f"| {n} | `{c['path']}` | {c['score']:.4g} | {c['margin']:+.3g} | "
                     f"{c['status']} | {_pct(c['area']) if c['area'] > 0 else '-'} |")
    return lines


def entry_lookup(entries: list[FrozenEntry], ratio: float) -> dict[tuple[str, str], FrozenEntry]:
    return {(e.model, e.category): e for e in entries if e.ratio == ratio}
