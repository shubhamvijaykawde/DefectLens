"""Error gallery: case ranking, box-level error attribution and the figures (tiny fixtures)."""

import json
from pathlib import Path

import cv2
import numpy as np

from defectlens.eval.detection import Detections, GroundTruth
from defectlens.eval.gallery import (
    area_summary,
    best_by_calibration_auroc,
    class_breakdown,
    image_errors,
    plot_neu_cases,
    plot_visa_cases,
    rank_visa_cases,
    worst_images,
)
from defectlens.eval.thresholds import detection_counts
from detection_fixtures import random_detections


def test_rank_visa_cases_orders_worst_first_and_flags_outcomes() -> None:
    paths = [f"i{k}" for k in range(6)]
    scores = [0.2, 0.9, 0.6, 0.1, 0.8, 0.4]
    labels = [1, 1, 1, 0, 0, 0]
    areas = [0.01, 0.2, 0.05, 0, 0, 0]
    cases = rank_visa_cases(paths, scores, labels, areas, threshold=0.5, top_k=2)
    assert [c["path"] for c in cases["anomalous"]] == ["i0", "i2"]  # lowest scores first
    assert [c["status"] for c in cases["anomalous"]] == ["missed", "caught"]
    assert [c["path"] for c in cases["normal"]] == ["i4", "i5"]  # highest scores first
    assert [c["status"] for c in cases["normal"]] == ["false alarm", "ok"]
    assert cases["anomalous"][0]["margin"] == np.float64(0.2) - 0.5


def test_area_summary() -> None:
    s = area_summary([0.2, 0.9, 0.3, 0.1, 0.8], [1, 1, 1, 0, 0], [0.01, 0.2, 0.03, 0, 0], 0.5)
    assert (s["n_missed"], s["n_caught"], s["n_false_alarms"], s["n_normal"]) == (2, 1, 1, 2)
    assert s["median_area_missed"] == 0.02 and s["median_area_caught"] == 0.2


def test_image_errors_agree_with_the_shared_counts() -> None:
    dets, gts = random_detections(np.random.default_rng(11), 25)
    for thr in (0.1, 0.5, 0.8):
        errors = [image_errors(d, g, thr) for d, g in zip(dets, gts, strict=True)]
        c = detection_counts(dets, gts, thr)
        assert (sum(e.tp for e in errors), sum(e.fp for e in errors),
                sum(e.fn for e in errors)) == (c.tp, c.fp, c.fn)


def test_image_errors_marks_the_right_boxes() -> None:
    gt = GroundTruth.of([[0, 0, 10, 10], [50, 50, 60, 60]], [1, 2])
    det = Detections.of([[0, 0, 10, 10], [80, 80, 90, 90], [50, 50, 60, 60]], [0.9, 0.8, 0.7],
                        [1, 1, 1])  # 3rd has the wrong class
    e = image_errors(det, gt, 0.5)
    assert e.pred_is_tp.tolist() == [True, False, False]
    assert e.gt_missed.tolist() == [False, True]
    assert image_errors(det, gt, 0.85).fp == 0  # only the 0.9 box is kept


def test_worst_images_and_class_breakdown() -> None:
    gts = [GroundTruth.of([[0, 0, 10, 10]], [1]), GroundTruth.of([[0, 0, 10, 10]], [2]),
           GroundTruth.of([], [])]
    dets = [Detections.empty(), Detections.of([[0, 0, 10, 10]], [0.9], [2]),
            Detections.of([[5, 5, 9, 9]] * 2, [0.9, 0.8], [1, 1])]
    errors = [image_errors(d, g, 0.5) for d, g in zip(dets, gts, strict=True)]
    worst = worst_images(errors, ["a", "b", "c"], ratio=10, top_k=2)
    assert [w["name"] for w in worst] == ["a", "c"]  # a: 1 miss = 10; c: 2 FP = 2; b: 0
    table = {r["class"]: r for r in class_breakdown(errors, gts, {1: "x", 2: "y"})}
    # image a is a plain miss (nothing kept there); localisation misses are tested below
    assert table["x"] == {"class": "x", "n_gt": 1, "missed": 1, "missed_overlapped": 0,
                          "false_alarms": 2}
    assert table["y"] == {"class": "y", "n_gt": 1, "missed": 0, "missed_overlapped": 0,
                          "false_alarms": 0}


def test_localisation_miss_is_told_apart_from_a_plain_miss() -> None:
    gt = GroundTruth.of([[0, 0, 100, 100], [200, 200, 210, 210]], [1, 1])
    det = Detections.of([[0, 0, 100, 30]], [0.9], [1])  # IoU 0.3 with the first box only
    e = image_errors(det, gt, 0.5)
    assert e.gt_missed.tolist() == [True, True]
    assert e.gt_near.tolist() == [True, False]
    assert image_errors(det, gt, 0.95).gt_near.tolist() == [False, False]  # box no longer kept


def test_best_by_calibration_auroc(tmp_path: Path) -> None:
    for name, scores in (("m1", [0.1, 0.2, 0.9, 0.8]), ("m2", [0.9, 0.2, 0.1, 0.8])):
        d = tmp_path / f"{name}_cat"
        d.mkdir()
        (d / "calibration_scores.json").write_text(
            json.dumps({"scores": scores, "labels": [0, 0, 1, 1]})
        )
    assert best_by_calibration_auroc(["m2", "m1"], "cat", tmp_path) == "m1"


def test_figures_are_written(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    images = rng.random((4, 16, 16, 3)).astype(np.float32)
    maps = rng.random((4, 16, 16)).astype(np.float32)
    masks = np.zeros((4, 16, 16), dtype=bool)
    masks[0, 4:8, 4:8] = True
    cases = rank_visa_cases([f"p{i}" for i in range(4)], [0.3, 0.7, 0.2, 0.9], [1, 1, 0, 0],
                            masks.reshape(4, -1).mean(1), 0.5, top_k=2)
    out = tmp_path / "visa.png"
    plot_visa_cases(images, maps, masks, cases, 0.5, "t", out, columns=3)
    assert out.stat().st_size > 1000

    names = ["a.jpg", "b.jpg"]
    for n in names:
        cv2.imwrite(str(tmp_path / n), np.full((40, 40), 90, dtype=np.uint8))
    gts = [GroundTruth.of([[2, 2, 20, 20]], [1]), GroundTruth.of([[5, 5, 30, 30]], [2])]
    dets = [Detections.of([[30, 30, 39, 39]], [0.9], [1]), Detections.empty()]
    errors = [image_errors(d, g, 0.5) for d, g in zip(dets, gts, strict=True)]
    out2 = tmp_path / "neu.png"
    plot_neu_cases(tmp_path, names, errors, gts, worst_images(errors, names, 10, 2),
                   {1: "x", 2: "y"}, "t", out2, columns=2)
    assert out2.stat().st_size > 1000
