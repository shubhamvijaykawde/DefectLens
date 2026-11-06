import numpy as np

from defectlens.eval.confusion import confusion_matrix, normalize_rows
from defectlens.eval.detection import Detections, GroundTruth
from defectlens.eval.report import confusion_table, metrics_table

A = [0.0, 0.0, 10.0, 10.0]
B = [50.0, 50.0, 60.0, 60.0]
C = [100.0, 100.0, 110.0, 110.0]


def test_confusion_hand_computed() -> None:
    # 3 classes (ids 1..3), background index 3.
    gts = [GroundTruth.of([A, B, C], [1, 2, 3])]
    dets = [
        Detections.of(
            [A, B, [200, 200, 210, 210], [0, 0, 10, 10]],
            [0.9, 0.8, 0.7, 0.1],
            [1, 3, 2, 1],  # A ok; B predicted as class 3 (confusion); stray box; low score
        )
    ]
    cm = confusion_matrix(dets, gts, 3, iou_thr=0.5, score_thr=0.25)
    expected = np.zeros((4, 4), dtype=np.int64)
    expected[0, 0] = 1  # A: class 1 -> 1
    expected[1, 2] = 1  # B: true class 2 predicted as 3
    expected[2, 3] = 1  # C: missed
    expected[3, 1] = 1  # stray class-2 box on background
    np.testing.assert_array_equal(cm, expected)


def test_duplicate_detection_is_a_false_positive() -> None:
    gts = [GroundTruth.of([A], [1])]
    dets = [Detections.of([A, A], [0.9, 0.8], [1, 1])]
    cm = confusion_matrix(dets, gts, 1)
    np.testing.assert_array_equal(cm, [[1, 0], [1, 0]])


def test_empty_images_and_normalisation() -> None:
    cm = confusion_matrix([Detections.empty()], [GroundTruth.of([], [])], 2)
    assert cm.shape == (3, 3) and cm.sum() == 0
    n = normalize_rows(np.array([[1, 3], [0, 0]]))
    np.testing.assert_allclose(n, [[0.25, 0.75], [0.0, 0.0]])


def test_markdown_tables() -> None:
    cm = np.array([[1, 0], [2, 0]])
    assert "| background | 2 | 0 |" in confusion_table(cm, ["x"])
    t = metrics_table({"val": {"map50": 0.5}, "test": {}}, ["map50"])
    assert "| map50 | 0.5000 | n/a |" in t
