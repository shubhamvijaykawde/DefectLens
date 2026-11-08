"""Phase 6a: do the frozen Phase 4a thresholds still hold when the scores come from the API's path?

The thresholds were chosen on scores from the PyTorch scoring path of Phase 4a (``eval.scoring``:
the cuDNN default for the anomaly models, fp16 autocast for the detectors). The API scores with
ONNX Runtime. For every served model this counts, on the **selection split only**
(``threshold_calibration`` for VisA, ``val`` for NEU-DET; ``check_split`` refuses the others), how
many images end on the other side of their frozen threshold, for ORT with TF32 off and on.

The ORT side is the serving code itself (``defectlens.serve``: byte decoding, preprocessing, session
options, detector decoding), not a re-implementation, so a difference in the serving path would
show up here too.

* Anomaly products: an image flips if ``(PyTorch score >= t) != (ORT score >= t)``.
* Detectors: a threshold keeps boxes, not images, so the count is the number of images whose number
  of kept boxes (``score >= t``) differs, the images whose alarm status (at least one kept box)
  differs, and the TP / FP / FN / cost at the threshold under both paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from defectlens.eval.detection import Detections, GroundTruth
from defectlens.eval.thresholds import FrozenEntry, check_split, detection_counts
from defectlens.serve.config import ProductSpec, ServeSettings
from defectlens.serve.inference import (
    decode_image,
    prepare_detector_input,
    preprocess_visa,
    run_anomaly_onnx,
    run_detector_onnx,
)
from defectlens.serve.model_registry import ModelRegistry


def side_flips(
    ref: npt.ArrayLike, got: npt.ArrayLike, threshold: float, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Images on the other side of ``threshold`` (flagged = ``score >= threshold``).

    ``margin_*`` is the smallest ``|score - threshold| / |threshold|`` over the images: how close
    the nearest image sits to the threshold under each path.
    """
    r = np.asarray(ref, dtype=np.float64).ravel()
    g = np.asarray(got, dtype=np.float64).ravel()
    if r.shape != g.shape:
        msg = f"{r.shape} reference scores for {g.shape} others"
        raise ValueError(msg)
    flip = (r >= threshold) != (g >= threshold)
    scale = abs(threshold) or 1.0
    return {
        "threshold": float(threshold),
        "n_images": int(r.size),
        "flips": int(flip.sum()),
        "flipped": [
            (names[int(i)] if names is not None else int(i)) for i in np.flatnonzero(flip)
        ],
        "margin_ref": float(np.abs(r - threshold).min() / scale),
        "margin_got": float(np.abs(g - threshold).min() / scale),
    }


def score_difference(ref: npt.ArrayLike, got: npt.ArrayLike) -> dict[str, float]:
    r = np.asarray(ref, dtype=np.float64).ravel()
    g = np.asarray(got, dtype=np.float64).ravel()
    d = np.abs(r - g)
    return {
        "max_abs": float(d.max()),
        "max_rel": float((d / np.maximum(np.abs(r), 1e-6)).max()),
    }


def detector_effect(
    ref: Sequence[Detections], got: Sequence[Detections], gts: Sequence[GroundTruth],
    entry: FrozenEntry, iou: float,
) -> dict[str, Any]:
    """What a frozen box-score threshold does under two score paths on the same images."""
    t = entry.threshold
    kept_r = [int((d.scores >= t).sum()) for d in ref]
    kept_g = [int((d.scores >= t).sum()) for d in got]
    differs = sum(a != b for a, b in zip(kept_r, kept_g, strict=True))
    alarm = sum((a > 0) != (b > 0) for a, b in zip(kept_r, kept_g, strict=True))
    out: dict[str, Any] = {
        "threshold": float(t), "n_images": len(ref),
        "images_kept_count_differs": int(differs), "images_alarm_flips": int(alarm),
        "kept_boxes_ref": int(sum(kept_r)), "kept_boxes_got": int(sum(kept_g)),
    }
    for tag, dets in (("ref", ref), ("got", got)):
        c = detection_counts(dets, gts, t, iou)
        out[tag] = {"tp": c.tp, "fp": c.fp, "fn": c.fn, "cost": c.cost(entry.ratio)}
    return out


def _variant(spec: ProductSpec, tf32: bool) -> ProductSpec:
    return spec.model_copy(update={"tf32": tf32})


def ort_registry(settings: ServeSettings, product: str, spec: ProductSpec) -> ModelRegistry:
    """A registry that serves only ``product`` with ``spec`` (to try another TF32 setting)."""
    return ModelRegistry(settings.model_copy(update={"products": {product: spec}}))


def anomaly_consistency(
    settings: ServeSettings, product: str, spec: ProductSpec, ref_paths: Sequence[str],
    ref_scores: npt.ArrayLike, visa_root: Path, entries: Sequence[FrozenEntry],
) -> dict[str, Any]:
    """Flips of one VisA product on its selection split, for ORT TF32 off and on.

    ``ref_paths`` / ``ref_scores`` are the Phase 4a PyTorch scores (already checked against the
    scores saved at training time by ``eval.scoring.providers``).
    """
    check_split("selection", "visa", "threshold_calibration")
    ref = np.asarray(ref_scores, dtype=np.float64)
    out: dict[str, Any] = {"n_images": len(ref_paths), "variants": {}}
    for tf32 in (False, True):
        reg = ort_registry(settings, product, _variant(spec, tf32))
        loaded = reg.acquire(product)
        got = []
        for rel in ref_paths:
            bgr = decode_image((visa_root / rel).read_bytes(), color=True,
                               max_pixels=settings.max_pixels)
            score, _ = run_anomaly_onnx(loaded, preprocess_visa(bgr, loaded.input_hw))
            got.append(score)
        out["variants"][f"tf32_{'on' if tf32 else 'off'}"] = {
            "provider": loaded.provider,
            "score_difference": score_difference(ref, got),
            "by_ratio": {
                f"{e.ratio:g}": side_flips(ref, got, e.threshold, ref_paths) for e in entries
            },
        }
        reg.evict_all()
    return out


def detector_consistency(
    settings: ServeSettings, product: str, spec: ProductSpec, ref: Sequence[Detections],
    gts: Sequence[GroundTruth], names: Sequence[str], images_dir: Path,
    entries: Sequence[FrozenEntry], iou: float,
) -> dict[str, Any]:
    """Effect of ORT (TF32 off and on) on a detector's frozen thresholds, on NEU-DET ``val``."""
    check_split("selection", "neu_det", "val")
    out: dict[str, Any] = {"n_images": len(ref), "variants": {}}
    for tf32 in (False, True):
        reg = ort_registry(settings, product, _variant(spec, tf32))
        loaded = reg.acquire(product)
        got: list[Detections] = []
        for name in names:
            gray = decode_image((images_dir / name).read_bytes(), color=False,
                                max_pixels=settings.max_pixels)
            x, affine = prepare_detector_input(spec, settings, gray, loaded.input_hw[0])
            got.append(run_detector_onnx(loaded, spec, settings, x, affine,
                                         (gray.shape[0], gray.shape[1])))
        out["variants"][f"tf32_{'on' if tf32 else 'off'}"] = {
            "provider": loaded.provider,
            "mean_detections_ref": float(np.mean([len(d.scores) for d in ref])),
            "mean_detections_got": float(np.mean([len(d.scores) for d in got])),
            "by_ratio": {
                f"{e.ratio:g}": detector_effect(ref, got, gts, e, iou) for e in entries
            },
        }
        reg.evict_all()
    return out

