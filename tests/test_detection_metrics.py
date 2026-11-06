import contextlib
import io

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from defectlens.eval.detection import (
    Detections,
    GroundTruth,
    average_precision,
    box_iou,
    evaluate_detections,
    match_greedy,
    operating_point,
)

NAMES = {1: "a", 2: "b"}
BOX = [0.0, 0.0, 10.0, 10.0]


def test_box_iou_hand_computed() -> None:
    a = np.array([BOX])
    b = np.array([BOX, [5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 50 / 150, 0.0]])
    assert box_iou(np.zeros((0, 4)), b).shape == (0, 3)


def test_match_greedy_takes_best_unmatched_gt() -> None:
    iou = np.array([[0.9, 0.6], [0.8, 0.1]])  # det 0 takes gt 0; det 1 cannot reuse it
    assert match_greedy(iou, 0.5).tolist() == [0, -1]


def test_perfect_predictions_score_one() -> None:
    gts = [
        GroundTruth.of([BOX, [20, 20, 40, 50]], [1, 2]),
        GroundTruth.of([[5, 5, 25, 15]], [1]),
    ]
    dets = [Detections.of(g.boxes, [0.9, 0.8][: len(g.boxes)], g.labels) for g in gts]
    for agnostic in (False, True):
        m = evaluate_detections(dets, gts, NAMES, class_agnostic=agnostic)
        assert m["map50"] == 1.0
        assert m["map50_95"] == 1.0
        assert m["recall"] == m["recall_max"] == m["precision"] == 1.0
        assert m["fp_per_image"] == 0.0


def test_ap_hand_computed_with_false_positive() -> None:
    # Two images, one GT each. Ranked detections: TP(.9), FP(.8), TP(.7).
    # Precision at the two recall levels: 1.0 up to recall 0.5, then 2/3 up to recall 1.0.
    # 101-point AP = (51 * 1 + 50 * 2/3) / 101 = 253 / 303.
    gts = [GroundTruth.of([BOX], [1]), GroundTruth.of([BOX], [1])]
    dets = [
        Detections.of([BOX], [0.9], [1]),
        Detections.of([[20, 20, 30, 30], BOX], [0.8, 0.7], [1, 1]),
    ]
    ap = average_precision(dets, gts, 1)
    assert ap[0] == pytest.approx(253 / 303, abs=1e-12)
    op = operating_point(dets, gts, 1, 0.5, 0.75)  # keeps only .9 and .8
    assert (op.tp, op.fp, op.fn) == (1, 1, 1)
    assert op.recall == 0.5
    assert op.precision == 0.5


def test_ap_over_iou_thresholds() -> None:
    # One GT, one detection with IoU 0.72: it matches thresholds .50-.70 (5 of 10), so AP = 0.5.
    gts = [GroundTruth.of([BOX], [1])]
    dets = [Detections.of([[0.0, 0.0, 10.0, 7.2]], [0.9], [1])]
    m = evaluate_detections(dets, gts, {1: "a"})
    assert m["map50"] == 1.0
    assert m["map50_95"] == pytest.approx(0.5)


def test_wrong_class_and_no_predictions() -> None:
    gts = [GroundTruth.of([BOX], [1])]
    wrong = [Detections.of([BOX], [0.9], [2])]
    m = evaluate_detections(wrong, gts, NAMES)
    assert m["ap50_a"] == 0.0 and m["recall_a"] == 0.0 and m["map50"] == 0.0
    assert np.isnan(m["ap50_b"])  # no ground truth for class b: excluded from the mean
    agnostic = evaluate_detections(wrong, gts, NAMES, class_agnostic=True)
    assert agnostic["map50"] == 1.0
    empty = evaluate_detections([Detections.empty()], gts, NAMES)
    assert empty["map50"] == 0.0 and empty["recall"] == 0.0


def test_recall_score_threshold_and_ceiling() -> None:
    gts = [GroundTruth.of([BOX], [1])]
    dets = [Detections.of([BOX], [0.3], [1])]
    m = evaluate_detections(dets, gts, {1: "a"}, recall_score_thr=0.5)
    assert m["recall"] == 0.0
    assert m["recall_max"] == 1.0


def test_no_ground_truth_raises() -> None:
    with pytest.raises(ValueError, match="no ground-truth"):
        evaluate_detections([Detections.empty()], [GroundTruth.of([], [])], NAMES)


def _random_scene(
    rng: np.random.Generator,
) -> tuple[list[Detections], list[GroundTruth]]:
    dets, gts = [], []
    for _ in range(40):
        n = int(rng.integers(0, 5))
        xy = rng.uniform(0, 150, (n, 2))
        wh = rng.uniform(15, 50, (n, 2))
        gb = np.hstack([xy, xy + wh])
        gl = rng.integers(1, 3, n)
        keep = rng.random(n) < 0.8  # some misses
        jitter = gb[keep] + rng.normal(0, 4, (int(keep.sum()), 4))
        fx = rng.uniform(0, 150, (3, 2))
        fp = np.hstack([fx, fx + rng.uniform(15, 50, (3, 2))])
        boxes = np.vstack([jitter, fp])
        labels = np.concatenate([gl[keep], rng.integers(1, 3, 3)])
        dets.append(Detections.of(boxes, rng.random(len(boxes)), labels))
        gts.append(GroundTruth.of(gb, gl))
    return dets, gts


def test_matches_pycocotools() -> None:
    """The metric code agrees with the reference COCO implementation on random data."""
    dets, gts = _random_scene(np.random.default_rng(0))
    images = [{"id": i + 1, "width": 200, "height": 200} for i in range(len(gts))]
    anns, results = [], []
    for i, (d, g) in enumerate(zip(dets, gts, strict=True), start=1):
        for b, lab in zip(g.boxes, g.labels, strict=True):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(lab),
                         "bbox": [b[0], b[1], w, h], "area": w * h, "iscrowd": 0})
        for b, s, lab in zip(d.boxes, d.scores, d.labels, strict=True):
            results.append({"image_id": i, "category_id": int(lab), "score": float(s),
                            "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})
    gt_api = COCO()
    gt_api.dataset = {"images": images, "annotations": anns,
                      "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt_api.createIndex()
        dt_api = gt_api.loadRes(results)
        ev = COCOeval(gt_api, dt_api, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    ours = evaluate_detections(dets, gts, NAMES)
    assert ours["map50_95"] == pytest.approx(ev.stats[0], abs=1e-9)
    assert ours["map50"] == pytest.approx(ev.stats[1], abs=1e-9)
