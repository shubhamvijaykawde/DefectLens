from pathlib import Path

import numpy as np
import pytest

from defectlens.config import load_config
from defectlens.data.coco import read_coco, to_coco, write_coco
from defectlens.data.neu_det import Box, Record
from defectlens.detection.classical import (
    DEFECT,
    ClassicalParams,
    detect,
    tune,
    z_map,
)
from defectlens.eval.detection import GroundTruth, box_iou


def _scene(seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """Textured gray steel with one rough patch at [60:100, 80:140] (x 80..140, y 60..100)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(120, 2, (200, 200))
    img[60:100, 80:140] += rng.normal(0, 25, (40, 60))
    return np.clip(img, 0, 255).astype(np.uint8), [80.0, 60.0, 140.0, 100.0]


def test_z_map_is_deterministic_and_highest_on_the_patch() -> None:
    img, _ = _scene()
    z1, z2 = z_map(img, 16.0, 2.0), z_map(img, 16.0, 2.0)
    np.testing.assert_array_equal(z1, z2)
    assert z1[65:95, 90:130].mean() > z1[:40].mean() + 2


def test_detects_the_patch() -> None:
    img, box = _scene()
    dets = detect(img, ClassicalParams(bg_sigma=16, smooth_sigma=3, z_thresh=2.0, min_area=200))
    assert len(dets.boxes) >= 1
    top = dets.boxes[int(np.argmax(dets.scores))]
    assert box_iou(top[None], np.array([box]))[0, 0] > 0.3  # smoothing makes the box a bit loose
    assert (dets.labels == DEFECT).all()


def test_tune_picks_a_config_that_scores_and_is_val_only() -> None:
    scenes = [_scene(s) for s in range(3)]
    imgs = [s[0] for s in scenes]
    gts = [GroundTruth.of([s[1]], [5]) for s in scenes]  # labels are ignored (class-agnostic)
    grid = {"bg_sigma": [16.0], "smooth_sigma": [3.0], "z_thresh": [0.1, 2.0],
            "close_kernel": [5], "min_area": [200]}
    best, metrics, tried = tune(imgs, gts, grid)
    assert tried == 2
    assert metrics["map50"] > 0.5
    assert best.z_thresh == 2.0  # 0.1 floods the image with one huge component


def test_read_coco_roundtrip(tmp_path: Path) -> None:
    rec = Record("crazing_1", "crazing_1.jpg", 200, 200, (Box("crazing", 10, 20, 50, 80),))
    write_coco(to_coco([rec]), tmp_path / "c.json")
    images, names = read_coco(tmp_path / "c.json")
    assert names[1] == "crazing"
    np.testing.assert_array_equal(images[0].boxes, [[10, 20, 50, 80]])
    assert images[0].labels.tolist() == [1]


def test_load_config_composes_model() -> None:
    cfg = load_config(["model=classical", "seed=7"])
    assert cfg.seed == 7 and cfg.model.name == "classical" and cfg.data.name == "neu_det"
    assert load_config().get("model") is None
    with pytest.raises(Exception):  # noqa: B017, PT011 (hydra raises its own error type)
        load_config(["model=does_not_exist"])


def test_read_predictions_aligns_with_images(tmp_path: Path) -> None:
    from defectlens.data.coco import read_predictions

    recs = [
        Record("crazing_1", "crazing_1.jpg", 200, 200, (Box("crazing", 10, 20, 50, 80),)),
        Record("crazing_2", "crazing_2.jpg", 200, 200, (Box("crazing", 1, 2, 5, 8),)),
    ]
    write_coco(to_coco(recs), tmp_path / "c.json")
    images, _ = read_coco(tmp_path / "c.json")
    preds = tmp_path / "p.json"
    preds.write_text('[{"image_id": 2, "category_id": 1, "score": 0.5, "bbox": [1, 2, 4, 6]}]')
    dets = read_predictions(preds, images)
    assert len(dets[0].boxes) == 0
    np.testing.assert_array_equal(dets[1].boxes, [[1, 2, 5, 8]])
