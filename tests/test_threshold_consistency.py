"""Counting images that change side of a frozen threshold between two score paths (Phase 6a)."""

from __future__ import annotations

import numpy as np
import pytest

from defectlens.eval.detection import Detections, GroundTruth
from defectlens.eval.thresholds import FrozenEntry
from defectlens.export.threshold_consistency import (
    detector_effect,
    score_difference,
    side_flips,
)


def test_side_flips_counts_images_that_cross_and_names_them() -> None:
    ref = [0.10, 0.49, 0.51, 0.90]
    got = [0.10, 0.51, 0.51, 0.40]   # image 1 crosses up, image 3 crosses down
    r = side_flips(ref, got, 0.5, names=["a", "b", "c", "d"])
    assert r["flips"] == 2 and r["flipped"] == ["b", "d"] and r["n_images"] == 4
    assert r["margin_ref"] == pytest.approx(0.01 / 0.5) and r["margin_got"] == pytest.approx(0.02)


def test_a_score_equal_to_the_threshold_is_flagged_in_both_paths() -> None:
    assert side_flips([0.5], [0.5], 0.5)["flips"] == 0
    assert side_flips([0.5], [0.4999999], 0.5)["flips"] == 1


def test_side_flips_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="reference"):
        side_flips([1.0, 2.0], [1.0], 0.5)


def test_score_difference() -> None:
    d = score_difference([1.0, 2.0], [1.0, 2.2])
    assert d["max_abs"] == pytest.approx(0.2) and d["max_rel"] == pytest.approx(0.1)


def _dets(scores: list[float]) -> Detections:
    n = len(scores)
    boxes = np.tile([0.0, 0.0, 10.0, 10.0], (n, 1))
    return Detections.of(boxes, scores, np.ones(n, dtype=int))


def test_detector_effect_counts_kept_box_changes_and_cost() -> None:
    gts = [GroundTruth.of([[0, 0, 10, 10]], [1]), GroundTruth.of([[0, 0, 10, 10]], [1])]
    ref = [_dets([0.9, 0.45]), _dets([0.8])]
    got = [_dets([0.9, 0.55]), _dets([0.8])]  # image 0 keeps one more box at t = 0.5
    entry = FrozenEntry("neu_det", "m", "all", 10.0, 0.5, "val", 2, 0, 0, 0, None, 0.0)
    r = detector_effect(ref, got, gts, entry, iou=0.5)
    assert r["images_kept_count_differs"] == 1 and r["images_alarm_flips"] == 0
    assert (r["kept_boxes_ref"], r["kept_boxes_got"]) == (2, 3)
    assert r["ref"] == {"tp": 2, "fp": 0, "fn": 0, "cost": 0.0}
    assert r["got"]["fp"] == 1 and r["got"]["cost"] == 1.0


def test_detector_effect_sees_an_image_that_loses_its_last_box() -> None:
    gts = [GroundTruth.of([[0, 0, 10, 10]], [1])]
    entry = FrozenEntry("neu_det", "m", "all", 5.0, 0.5, "val", 1, 0, 0, 0, None, 0.0)
    r = detector_effect([_dets([0.6])], [_dets([0.4])], gts, entry, iou=0.5)
    assert r["images_alarm_flips"] == 1 and r["got"]["fn"] == 1 and r["got"]["cost"] == 5.0
