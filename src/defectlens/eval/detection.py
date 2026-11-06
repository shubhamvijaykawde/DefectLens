"""Detection metrics: COCO-style AP (class-aware or class-agnostic), recall and precision.

Boxes are ``[x1, y1, x2, y2]`` in the original image's pixels. The AP follows the COCO protocol
(greedy matching in descending score order, 101-point interpolated precision, at most ``max_dets``
detections per image and class, all box sizes) and is cross-checked against ``pycocotools`` in
``tests/test_detection_metrics.py``. Every detector in this repo is scored by this one module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

AP_IOU_THRESHOLDS: FloatArray = np.round(np.arange(0.5, 0.951, 0.05), 2)
RECALL_THRESHOLDS: FloatArray = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class Detections:
    """Predictions for one image."""

    boxes: FloatArray  # (N, 4) xyxy
    scores: FloatArray  # (N,)
    labels: IntArray  # (N,)

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64))

    @classmethod
    def of(cls, boxes: npt.ArrayLike, scores: npt.ArrayLike, labels: npt.ArrayLike) -> Detections:
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        return cls(b, np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64))


@dataclass(frozen=True)
class GroundTruth:
    """Annotations for one image."""

    boxes: FloatArray  # (M, 4) xyxy
    labels: IntArray  # (M,)

    @classmethod
    def of(cls, boxes: npt.ArrayLike, labels: npt.ArrayLike) -> GroundTruth:
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        return cls(b, np.asarray(labels, dtype=np.int64))


def box_iou(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pairwise IoU of ``(N, 4)`` and ``(M, 4)`` xyxy boxes -> ``(N, M)``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def match_greedy(iou: FloatArray, thr: float) -> IntArray:
    """Greedy one-to-one matching. Rows (detections) must already be in descending score order.

    Each detection takes the unmatched ground truth with the highest IoU if that IoU is at
    least ``thr``. Returns, per detection, the matched ground-truth index or -1.
    """
    n_det, n_gt = iou.shape
    match = np.full(n_det, -1, dtype=np.int64)
    taken = np.zeros(n_gt, dtype=bool)
    # COCO compares against min(thr, 1 - 1e-10) so that thr = 1.0 can still match identical boxes.
    eff = min(thr, 1.0 - 1e-10)
    for d in range(n_det):
        if n_gt == 0:
            break
        cand = np.where(taken, -1.0, iou[d])
        g = int(np.argmax(cand))
        if cand[g] >= eff:
            match[d] = g
            taken[g] = True
    return match


def _agnostic(
    dets: Sequence[Detections], gts: Sequence[GroundTruth]
) -> tuple[list[Detections], list[GroundTruth]]:
    d = [Detections(x.boxes, x.scores, np.zeros_like(x.labels)) for x in dets]
    g = [GroundTruth(x.boxes, np.zeros_like(x.labels)) for x in gts]
    return d, g


def _check(dets: Sequence[Detections], gts: Sequence[GroundTruth]) -> None:
    if len(dets) != len(gts):
        msg = f"{len(dets)} detection sets for {len(gts)} ground-truth sets"
        raise ValueError(msg)


def average_precision(
    dets: Sequence[Detections],
    gts: Sequence[GroundTruth],
    label: int,
    iou_thresholds: FloatArray = AP_IOU_THRESHOLDS,
    max_dets: int = 100,
) -> FloatArray:
    """AP of one class at each IoU threshold. ``nan`` where the class has no ground truth."""
    _check(dets, gts)
    n_t = len(iou_thresholds)
    scores: list[FloatArray] = []
    tp_cols: list[npt.NDArray[np.bool_]] = []
    n_gt = 0
    for det, gt in zip(dets, gts, strict=True):
        dsel = np.flatnonzero(det.labels == label)
        order = dsel[np.argsort(-det.scores[dsel], kind="stable")][:max_dets]
        gboxes = gt.boxes[gt.labels == label]
        n_gt += len(gboxes)
        if len(order) == 0:
            continue
        iou = box_iou(det.boxes[order], gboxes)
        tp = np.zeros((n_t, len(order)), dtype=bool)
        for ti, thr in enumerate(iou_thresholds):
            tp[ti] = match_greedy(iou, float(thr)) >= 0
        scores.append(det.scores[order])
        tp_cols.append(tp)
    if n_gt == 0:
        return np.full(n_t, np.nan)
    if not scores:
        return np.zeros(n_t)
    all_scores = np.concatenate(scores)
    all_tp = np.concatenate(tp_cols, axis=1)
    order = np.argsort(-all_scores, kind="stable")
    all_tp = all_tp[:, order]
    ap = np.zeros(n_t)
    for ti in range(n_t):
        tps = np.cumsum(all_tp[ti])
        fps = np.cumsum(~all_tp[ti])
        recall = tps / n_gt
        precision = tps / (tps + fps)  # at least one detection, so never 0/0
        precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope
        idx = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
        valid = idx < len(precision)
        q = np.zeros(len(RECALL_THRESHOLDS))
        q[valid] = precision[idx[valid]]
        ap[ti] = q.mean()
    return ap


@dataclass(frozen=True)
class OperatingPoint:
    """Counts at one IoU threshold and one score threshold."""

    tp: int
    fp: int
    fn: int

    @property
    def recall(self) -> float:
        n = self.tp + self.fn
        return self.tp / n if n else float("nan")

    @property
    def precision(self) -> float:
        n = self.tp + self.fp
        return self.tp / n if n else float("nan")


def operating_point(
    dets: Sequence[Detections],
    gts: Sequence[GroundTruth],
    label: int | None,
    iou_thr: float = 0.5,
    score_thr: float = 0.0,
) -> OperatingPoint:
    """TP/FP/FN for one class (or ``label=None``: every class, matched class-aware)."""
    _check(dets, gts)
    tp = fp = fn = 0
    for det, gt in zip(dets, gts, strict=True):
        labels = np.unique(np.concatenate([det.labels, gt.labels])) if label is None else [label]
        for c in labels:
            dsel = np.flatnonzero((det.labels == c) & (det.scores >= score_thr))
            order = dsel[np.argsort(-det.scores[dsel], kind="stable")]
            gboxes = gt.boxes[gt.labels == c]
            m = match_greedy(box_iou(det.boxes[order], gboxes), iou_thr)
            hits = int((m >= 0).sum())
            tp += hits
            fp += len(order) - hits
            fn += len(gboxes) - hits
    return OperatingPoint(tp, fp, fn)


def evaluate_detections(
    dets: Sequence[Detections],
    gts: Sequence[GroundTruth],
    class_names: Mapping[int, str],
    *,
    class_agnostic: bool = False,
    max_dets: int = 100,
    recall_iou: float = 0.5,
    recall_score_thr: float = 0.5,
) -> dict[str, float]:
    """All headline numbers for one split, as a flat dict of floats.

    Keys: ``map50``, ``map50_95`` (mean over classes that have ground truth), ``ap50_<class>``,
    ``recall`` / ``precision`` / ``fp_per_image`` at ``recall_score_thr``, ``recall_<class>`` at the
    same point, and ``recall_max`` (no score cut-off: the ceiling set by the detector's proposals).
    With ``class_agnostic=True`` every box is one class named ``defect``.
    """
    _check(dets, gts)
    if class_agnostic:
        dets, gts = _agnostic(dets, gts)
        class_names = {0: "defect"}
    out: dict[str, float] = {}
    aps = {c: average_precision(dets, gts, c, max_dets=max_dets) for c in class_names}
    present = [c for c in class_names if not np.isnan(aps[c][0])]
    if not present:
        msg = "no ground-truth boxes in this split"
        raise ValueError(msg)
    out["map50"] = float(np.mean([aps[c][0] for c in present]))
    out["map50_95"] = float(np.mean([aps[c].mean() for c in present]))
    for c, name in class_names.items():
        out[f"ap50_{name}"] = float(aps[c][0])
        out[f"recall_{name}"] = operating_point(
            dets, gts, c, recall_iou, recall_score_thr
        ).recall
    at = operating_point(dets, gts, None, recall_iou, recall_score_thr)
    out["recall"] = at.recall
    out["precision"] = at.precision
    out["fp_per_image"] = at.fp / max(len(gts), 1)
    out["recall_max"] = operating_point(dets, gts, None, recall_iou, -np.inf).recall
    return out
