"""Classical, class-agnostic defect detector (no learning, a handful of tuned parameters).

Pipeline per grayscale image:

1. residual = image - Gaussian background (large sigma), so slow illumination changes cancel;
2. energy = Gaussian-smoothed ``|residual|``;
3. z = robust z-score of the energy (median / MAD of this image);
4. mask = z > ``z_thresh``, closed with an elliptical kernel;
5. each connected component with at least ``min_area`` pixels becomes a box, scored by its mean z.

It outputs boxes and scores only, with a single class, so it is compared with the deep models on
class-agnostic AP and recall, not per-class mAP. Parameters are chosen by :func:`tune` on the
validation split only.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import numpy.typing as npt

from defectlens.eval.detection import Detections, GroundTruth, evaluate_detections

Gray = npt.NDArray[np.uint8]
ZMap = npt.NDArray[np.float32]
DEFECT = 1  # the single class id used by this detector


@dataclass(frozen=True)
class ClassicalParams:
    bg_sigma: float = 16.0
    smooth_sigma: float = 2.0
    z_thresh: float = 1.5
    close_kernel: int = 5
    min_area: int = 50


def z_map(gray: Gray, bg_sigma: float, smooth_sigma: float) -> ZMap:
    """Steps 1-3: robust z-score of the local residual energy."""
    img = gray.astype(np.float32)
    resid = np.abs(img - cv2.GaussianBlur(img, (0, 0), bg_sigma))
    energy = cv2.GaussianBlur(resid, (0, 0), smooth_sigma)
    med = float(np.median(energy))
    mad = float(np.median(np.abs(energy - med)))
    z: ZMap = ((energy - med) / (1.4826 * mad + 1e-6)).astype(np.float32)
    return z


def boxes_from_z(z: ZMap, z_thresh: float, close_kernel: int, min_area: int) -> Detections:
    """Steps 4-5: threshold, close, label components, emit boxes."""
    mask = (z > z_thresh).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    mask = np.asarray(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), dtype=np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes, scores = [], []
    for i in range(1, n):  # 0 is background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append([x, y, x + w, y + h])
        scores.append(float(z[labels == i].mean()))
    return Detections.of(boxes, scores, np.full(len(boxes), DEFECT))


def detect(gray: Gray, params: ClassicalParams) -> Detections:
    return boxes_from_z(
        z_map(gray, params.bg_sigma, params.smooth_sigma),
        params.z_thresh,
        params.close_kernel,
        params.min_area,
    )


def agnostic_gt(gts: Sequence[GroundTruth]) -> list[GroundTruth]:
    """Ground truth with every label set to ``DEFECT`` (this detector has no class output)."""
    return [GroundTruth(g.boxes, np.full(len(g.boxes), DEFECT, dtype=np.int64)) for g in gts]


def score(dets: Sequence[Detections], gts: Sequence[GroundTruth]) -> dict[str, float]:
    """Class-agnostic metrics. Every emitted box counts (score cut-off 0): the detector's own
    ``z_thresh`` is its operating point."""
    return evaluate_detections(
        dets, agnostic_gt(gts), {DEFECT: "defect"}, class_agnostic=True, recall_score_thr=0.0
    )


def tune(
    images: Sequence[Gray],
    gts: Sequence[GroundTruth],
    grid: Mapping[str, Sequence[float]],
) -> tuple[ClassicalParams, dict[str, float], int]:
    """Exhaustive search maximising ``(map50, map50_95)`` on the given (validation) images.

    Returns the best parameters, their metrics and the number of configurations tried. The
    z-maps are cached per ``(bg_sigma, smooth_sigma)``; ties keep the earliest configuration.
    """
    best: tuple[tuple[float, float], ClassicalParams, dict[str, float]] | None = None
    tried = 0
    for bg, sm in itertools.product(grid["bg_sigma"], grid["smooth_sigma"]):
        zs = [z_map(im, float(bg), float(sm)) for im in images]
        for zt, ck, ma in itertools.product(
            grid["z_thresh"], grid["close_kernel"], grid["min_area"]
        ):
            p = ClassicalParams(float(bg), float(sm), float(zt), int(ck), int(ma))
            dets = [boxes_from_z(z, p.z_thresh, p.close_kernel, p.min_area) for z in zs]
            m = score(dets, gts)
            tried += 1
            key = (m["map50"], m["map50_95"])
            if best is None or key > best[0]:
                best = (key, p, m)
    if best is None:
        msg = "empty grid"
        raise ValueError(msg)
    return best[1], best[2], tried


def params_dict(p: ClassicalParams) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(p).items()}
