"""Anomaly metrics: hand-computed values, and AUPRO against anomalib's implementation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from defectlens.eval.anomaly import aupro, auroc, evaluate_anomaly, save_heatmap_grid


def test_auroc_textbook_example() -> None:
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)


def test_auroc_perfect_inverted_and_ties() -> None:
    assert auroc([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert auroc([1, 1, 0, 0], [0, 0, 1, 1]) == 0.0
    assert auroc([5, 5, 5, 5], [0, 1, 0, 1]) == 0.5


def test_auroc_single_class_is_nan() -> None:
    assert np.isnan(auroc([0.2, 0.9], [1, 1]))


def _masks() -> np.ndarray:
    m = np.zeros((6, 32, 32), dtype=bool)
    m[0, 4:12, 4:12] = True
    m[0, 20:26, 20:30] = True
    m[1, 10:20, 5:15] = True
    m[2, 0:6, 0:6] = True
    return m


def test_aupro_perfect_map_is_one() -> None:
    masks = _masks()
    assert aupro(masks.astype(np.float32), masks) == pytest.approx(1.0)


def test_aupro_degenerate_inputs_are_nan() -> None:
    maps = np.random.default_rng(0).random((2, 8, 8))
    assert np.isnan(aupro(maps, np.zeros((2, 8, 8), dtype=bool)))
    assert np.isnan(aupro(np.ones((2, 8, 8)), np.ones((2, 8, 8), dtype=bool)))
    with pytest.raises(ValueError, match="differ"):
        aupro(maps, np.zeros((2, 8, 9), dtype=bool))


@pytest.mark.parametrize("shift", [1.5, 0.5])
def test_aupro_matches_anomalib(shift: float) -> None:
    from anomalib.metrics import AUPRO

    rng = np.random.default_rng(0)
    masks = _masks()
    maps = (rng.normal(size=masks.shape) + masks * shift).astype(np.float32)

    class Batch:
        anomaly_map = torch.from_numpy(maps)
        gt_mask = torch.from_numpy(masks.astype(np.int64))

    ref = AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=0.3)
    ref.update(Batch())
    assert aupro(maps, masks) == pytest.approx(float(ref.compute()), abs=1e-3)


def test_evaluate_anomaly_and_heatmap_grid(tmp_path: Path) -> None:
    masks = _masks()
    maps = masks.astype(np.float32)
    labels = masks.reshape(6, -1).any(axis=1).astype(np.int64)
    scores = maps.reshape(6, -1).max(axis=1)
    out = evaluate_anomaly(scores, labels, maps, masks)
    assert out == {"image_auroc": 1.0, "pixel_auroc": 1.0, "aupro": pytest.approx(1.0)}
    images = np.zeros((6, 32, 32, 3), dtype=np.float32)
    path = tmp_path / "g" / "heat.png"
    save_heatmap_grid(images, maps, masks, scores, labels, path, "t", n_per_kind=2)
    assert path.stat().st_size > 0
