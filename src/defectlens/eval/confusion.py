"""Detection confusion matrix with an explicit background row and column."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from defectlens.eval.detection import Detections, GroundTruth, box_iou


def confusion_matrix(
    dets: Sequence[Detections],
    gts: Sequence[GroundTruth],
    num_classes: int,
    *,
    iou_thr: float = 0.5,
    score_thr: float = 0.25,
) -> npt.NDArray[np.int64]:
    """``(K+1, K+1)`` counts; rows are the true class, columns the predicted class.

    Labels ``1..K`` map to indices ``0..K-1``; index ``K`` is background. Detections below
    ``score_thr`` are ignored. Each detection (highest score first) is matched to the unmatched
    ground truth of **any** class with the highest IoU >= ``iou_thr``. Then:

    * matched -> ``cm[true, predicted]`` (off-diagonal means a class confusion),
    * ground truth left unmatched -> ``cm[true, K]`` (missed defect),
    * detection left unmatched -> ``cm[K, predicted]`` (false alarm on background).
    """
    if len(dets) != len(gts):
        msg = f"{len(dets)} detection sets for {len(gts)} ground-truth sets"
        raise ValueError(msg)
    k = num_classes
    cm = np.zeros((k + 1, k + 1), dtype=np.int64)
    for det, gt in zip(dets, gts, strict=True):
        keep = np.flatnonzero(det.scores >= score_thr)
        order = keep[np.argsort(-det.scores[keep], kind="stable")]
        iou = box_iou(det.boxes[order], gt.boxes)
        taken = np.zeros(len(gt.boxes), dtype=bool)
        for row, d in enumerate(order):
            pred = int(det.labels[d]) - 1
            cand = np.where(taken, -1.0, iou[row]) if len(taken) else np.array([-1.0])
            g = int(np.argmax(cand))
            if len(taken) and cand[g] >= iou_thr:
                taken[g] = True
                cm[int(gt.labels[g]) - 1, pred] += 1
            else:
                cm[k, pred] += 1
        for missed in np.flatnonzero(~taken):
            cm[int(gt.labels[missed]) - 1, k] += 1
    return cm


def normalize_rows(cm: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Row-normalise (each row sums to 1; all-zero rows stay zero)."""
    totals = cm.sum(axis=1, keepdims=True)
    out: npt.NDArray[np.float64] = np.divide(cm, totals, out=np.zeros(cm.shape), where=totals > 0)
    return out
