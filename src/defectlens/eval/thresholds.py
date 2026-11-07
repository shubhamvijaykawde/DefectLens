"""Cost-based operating points and the selection / evaluation protocol around them.

Cost model: ``cost = fn_fp_ratio * FN + FP`` (a miss costs ``fn_fp_ratio`` false alarms).

* VisA (image level): an image is flagged when ``score >= threshold``. FN = anomalous image not
  flagged, FP = normal image flagged.
* NEU-DET (box level): a detection is kept when ``score >= threshold`` (one threshold per model,
  for every class). TP/FP/FN come from the shared greedy IoU matching of ``eval/detection.py``
  (class-aware); FN = ground-truth box not matched, FP = kept box that matched nothing.

Protocol, enforced in code (``check_split``): a threshold is *selected* only on
``threshold_calibration`` (VisA) or ``val`` (NEU-DET) and *evaluated* only on
``threshold_evaluation`` (VisA) or ``test`` (NEU-DET), from a frozen file whose content hash is
verified before any evaluation data is read.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from defectlens.eval.detection import (
    Detections,
    GroundTruth,
    box_iou,
    match_greedy,
    operating_point,
)

SELECTION_SPLITS = {"visa": "threshold_calibration", "neu_det": "val"}
EVALUATION_SPLITS = {"visa": "threshold_evaluation", "neu_det": "test"}
FROZEN_VERSION = 1

FloatArr = npt.NDArray[np.float64]


class ProtocolError(RuntimeError):
    """A step tried to read data or use a file that the threshold protocol forbids."""


def check_split(purpose: str, domain: str, split: str) -> None:
    """Allow a selection step only on its selection split and evaluation only on its own."""
    tables = {"selection": SELECTION_SPLITS, "evaluation": EVALUATION_SPLITS}
    if purpose not in tables:
        msg = f"unknown purpose {purpose!r}, expected 'selection' or 'evaluation'"
        raise ValueError(msg)
    if domain not in SELECTION_SPLITS:
        msg = f"unknown domain {domain!r}, expected one of {sorted(SELECTION_SPLITS)}"
        raise ValueError(msg)
    allowed = tables[purpose][domain]
    if split != allowed:
        msg = f"{purpose} may only read {allowed!r} for {domain}, got {split!r}"
        raise ProtocolError(msg)


def cost(fn: int, fp: int, ratio: float) -> float:
    """``ratio * FN + FP``."""
    return float(ratio * fn + fp)


@dataclass(frozen=True)
class Counts:
    """Confusion counts at one threshold (``tn`` is None for box-level detection)."""

    tp: int
    fp: int
    fn: int
    tn: int | None = None

    def cost(self, ratio: float) -> float:
        return cost(self.fn, self.fp, ratio)

    @property
    def recall(self) -> float:
        n = self.tp + self.fn
        return self.tp / n if n else float("nan")

    @property
    def precision(self) -> float:
        n = self.tp + self.fp
        return self.tp / n if n else float("nan")

    @property
    def false_positive_rate(self) -> float:
        if self.tn is None:
            return float("nan")
        n = self.fp + self.tn
        return self.fp / n if n else float("nan")


@dataclass(frozen=True)
class Selection:
    """The chosen threshold with its counts on the data it was chosen on."""

    threshold: float
    counts: Counts
    ratio: float
    n_items: int  # images (VisA) or ground-truth boxes (NEU-DET)

    @property
    def cost(self) -> float:
        return self.counts.cost(self.ratio)


def candidate_thresholds(scores: npt.ArrayLike) -> FloatArr:
    """Every distinct decision for ``score >= t``, as finite thresholds, ascending.

    The first flags everything, the last flags nothing, the rest are midpoints between
    neighbouring distinct scores.
    """
    u = np.unique(np.asarray(scores, dtype=np.float64))
    if len(u) == 0:
        msg = "no scores to choose a threshold from"
        raise ValueError(msg)
    mids = (u[:-1] + u[1:]) / 2
    return np.concatenate([[np.nextafter(u[0], -np.inf)], mids, [np.nextafter(u[-1], np.inf)]])


def _pick(costs: FloatArr) -> int:
    """Index of the minimum cost; among ties the median tied candidate (deterministic)."""
    tied = np.flatnonzero(costs == costs.min())
    return int(tied[len(tied) // 2])


def flag_counts(scores: npt.ArrayLike, labels: npt.ArrayLike, threshold: float) -> Counts:
    """Image-level counts when images with ``score >= threshold`` are flagged anomalous."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    flag = s >= threshold
    return Counts(
        tp=int((flag & y).sum()), fp=int((flag & ~y).sum()),
        fn=int((~flag & y).sum()), tn=int((~flag & ~y).sum()),
    )


def select_binary(scores: npt.ArrayLike, labels: npt.ArrayLike, ratio: float) -> Selection:
    """Threshold minimising ``ratio * FN + FP`` over all image-level decisions."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if s.shape != y.shape:
        msg = f"{s.shape} scores for {y.shape} labels"
        raise ValueError(msg)
    if ratio <= 0:
        msg = f"fn_fp_ratio must be positive, got {ratio}"
        raise ValueError(msg)
    cands = candidate_thresholds(s)
    all_counts = [flag_counts(s, y, float(t)) for t in cands]
    best = _pick(np.array([c.cost(ratio) for c in all_counts]))
    return Selection(float(cands[best]), all_counts[best], float(ratio), len(s))


def detection_flags(
    dets: Sequence[Detections], gts: Sequence[GroundTruth], iou_thr: float = 0.5
) -> tuple[FloatArr, npt.NDArray[np.bool_], int]:
    """Score and true-positive flag of every detection (class-aware, greedy IoU matching), and
    the number of ground-truth boxes.

    Greedy matching walks detections in descending score order, so dropping the detections below
    a score threshold never changes the matches of the ones above it: the counts at any threshold
    are suffix sums of these flags (``tests/test_thresholds.py`` checks this against
    ``operating_point`` at every candidate).
    """
    if len(dets) != len(gts):
        msg = f"{len(dets)} detection sets for {len(gts)} ground-truth sets"
        raise ValueError(msg)
    scores: list[FloatArr] = []
    flags: list[npt.NDArray[np.bool_]] = []
    n_gt = 0
    for det, gt in zip(dets, gts, strict=True):
        n_gt += len(gt.labels)
        for c in np.unique(det.labels):
            sel = np.flatnonzero(det.labels == c)
            order = sel[np.argsort(-det.scores[sel], kind="stable")]
            m = match_greedy(box_iou(det.boxes[order], gt.boxes[gt.labels == c]), iou_thr)
            scores.append(det.scores[order])
            flags.append(m >= 0)
    if not scores:
        return np.zeros(0), np.zeros(0, dtype=bool), n_gt
    return np.concatenate(scores), np.concatenate(flags), n_gt


def select_detection(
    dets: Sequence[Detections], gts: Sequence[GroundTruth], ratio: float, iou_thr: float = 0.5
) -> Selection:
    """Score threshold minimising ``ratio * FN + FP`` over box-level decisions."""
    if ratio <= 0:
        msg = f"fn_fp_ratio must be positive, got {ratio}"
        raise ValueError(msg)
    scores, tp_flag, n_gt = detection_flags(dets, gts, iou_thr)
    cands = candidate_thresholds(scores)
    order = np.argsort(scores, kind="stable")
    s, tp = scores[order], tp_flag[order]
    tp_suffix = np.concatenate([np.cumsum(tp[::-1])[::-1], [0]])  # TPs with score >= s[i]
    idx = np.searchsorted(s, cands, side="left")
    tps = tp_suffix[idx]
    flagged = len(s) - idx
    all_counts = [
        Counts(tp=int(t), fp=int(f - t), fn=int(n_gt - t))
        for t, f in zip(tps, flagged, strict=True)
    ]
    best = _pick(np.array([c.cost(ratio) for c in all_counts]))
    return Selection(float(cands[best]), all_counts[best], float(ratio), n_gt)


def detection_counts(
    dets: Sequence[Detections], gts: Sequence[GroundTruth], threshold: float, iou_thr: float = 0.5
) -> Counts:
    """Box-level counts keeping detections with ``score >= threshold`` (shared matching code)."""
    op = operating_point(dets, gts, None, iou_thr, threshold)
    return Counts(tp=op.tp, fp=op.fp, fn=op.fn)


def trivial_cost(counts: Counts, ratio: float) -> float:
    """Cost of the best constant policy: flag nothing, or (image level only) flag everything."""
    miss_all = ratio * (counts.tp + counts.fn)
    if counts.tn is None:
        return float(miss_all)
    return float(min(miss_all, counts.fp + counts.tn))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n`` (NaN pair when ``n == 0``)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------------------------
# The frozen file


@dataclass(frozen=True)
class FrozenEntry:
    """One frozen threshold: a (domain, model, category, ratio) with its selection counts."""

    domain: str  # "visa" or "neu_det"
    model: str
    category: str  # VisA category, or "all" for NEU-DET (one threshold for every class)
    ratio: float
    threshold: float
    selection_split: str
    n_items: int
    tp: int
    fp: int
    fn: int
    tn: int | None
    cost: float

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.domain, self.model, self.category)


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def write_frozen(path: Path, entries: Sequence[FrozenEntry], meta: dict[str, Any]) -> str:
    """Write the frozen thresholds; returns the file's SHA-256.

    The file embeds ``content_sha256`` (over ``meta`` and ``entries``) so any later edit by hand
    is caught by ``read_frozen``; the returned value hashes the whole file as written.
    """
    for e in entries:
        check_split("selection", e.domain, e.selection_split)
    payload = {
        "meta": {**meta, "version": FROZEN_VERSION},
        "entries": [asdict(e) for e in entries],
    }
    doc = {**payload, "content_sha256": hashlib.sha256(_canonical(payload)).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return file_sha256(path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frozen(path: Path) -> tuple[list[FrozenEntry], dict[str, Any]]:
    """Load and verify the frozen file. Raises ``ProtocolError`` if it is missing or was edited."""
    if not path.is_file():
        msg = f"{path} does not exist: run scripts/select_thresholds.py first (evaluation needs it)"
        raise ProtocolError(msg)
    doc = json.loads(path.read_text())
    payload = {"meta": doc["meta"], "entries": doc["entries"]}
    if hashlib.sha256(_canonical(payload)).hexdigest() != doc.get("content_sha256"):
        msg = f"{path} does not match its embedded content hash: it was edited after freezing"
        raise ProtocolError(msg)
    entries = [FrozenEntry(**e) for e in doc["entries"]]
    for e in entries:
        check_split("selection", e.domain, e.selection_split)
    return entries, dict(doc["meta"])


# --------------------------------------------------------------------------------------------
# Building entries (selection) and scoring them (evaluation)


def entries_binary(
    model: str, category: str, split: str, scores: npt.ArrayLike, labels: npt.ArrayLike,
    ratios: Sequence[float],
) -> list[FrozenEntry]:
    """One frozen entry per ratio for an image-level model on its selection split."""
    check_split("selection", "visa", split)
    out = []
    for ratio in ratios:
        sel = select_binary(scores, labels, ratio)
        c = sel.counts
        again = flag_counts(scores, labels, sel.threshold)  # the evaluator's code path
        if again != c:
            msg = f"selection counts {c} differ from the evaluator's {again} at {sel.threshold}"
            raise ProtocolError(msg)
        out.append(FrozenEntry("visa", model, category, float(ratio), sel.threshold, split,
                               sel.n_items, c.tp, c.fp, c.fn, c.tn, sel.cost))
    return out


def entries_detection(
    model: str, split: str, dets: Sequence[Detections], gts: Sequence[GroundTruth],
    ratios: Sequence[float], iou_thr: float = 0.5,
) -> list[FrozenEntry]:
    """One frozen entry per ratio for a detector on its selection split."""
    check_split("selection", "neu_det", split)
    out = []
    for ratio in ratios:
        sel = select_detection(dets, gts, ratio, iou_thr)
        c = sel.counts
        again = detection_counts(dets, gts, sel.threshold, iou_thr)  # the evaluator's code path
        if again != c:
            msg = f"selection counts {c} differ from the evaluator's {again} at {sel.threshold}"
            raise ProtocolError(msg)
        out.append(FrozenEntry("neu_det", model, "all", float(ratio), sel.threshold, split,
                               sel.n_items, c.tp, c.fp, c.fn, None, sel.cost))
    return out


def evaluate_entry(
    entry: FrozenEntry, split: str, *, scores: npt.ArrayLike | None = None,
    labels: npt.ArrayLike | None = None, dets: Sequence[Detections] | None = None,
    gts: Sequence[GroundTruth] | None = None, iou_thr: float = 0.5,
) -> dict[str, Any]:
    """Counts and cost of one frozen threshold on its evaluation split (no tuning happens here).

    Pass ``scores`` and ``labels`` for a VisA entry, ``dets`` and ``gts`` for a NEU-DET entry.
    """
    check_split("evaluation", entry.domain, split)
    if entry.domain == "visa":
        if scores is None or labels is None:
            msg = "a visa entry needs scores and labels"
            raise ValueError(msg)
        counts = flag_counts(scores, labels, entry.threshold)
        n_items = len(np.asarray(scores).ravel())
    else:
        if dets is None or gts is None:
            msg = "a neu_det entry needs dets and gts"
            raise ValueError(msg)
        counts = detection_counts(dets, gts, entry.threshold, iou_thr)
        n_items = counts.tp + counts.fn
    row: dict[str, Any] = {
        "domain": entry.domain, "model": entry.model, "category": entry.category,
        "ratio": entry.ratio, "threshold": entry.threshold, "evaluation_split": split,
        "n_items": n_items, "tp": counts.tp, "fp": counts.fp, "fn": counts.fn, "tn": counts.tn,
        "cost": counts.cost(entry.ratio), "cost_per_item": counts.cost(entry.ratio) / n_items,
        "selection_cost_per_item": entry.cost / entry.n_items,
        "trivial_cost": trivial_cost(counts, entry.ratio),
        "recall": counts.recall, "precision": counts.precision,
        "false_positive_rate": counts.false_positive_rate,
    }
    if entry.domain == "visa":
        row["recall_ci"] = wilson_interval(counts.tp, counts.tp + counts.fn)
        row["false_positive_rate_ci"] = (
            wilson_interval(counts.fp, counts.fp + counts.tn) if counts.tn is not None else None
        )
    return row
