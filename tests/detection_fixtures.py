"""Random detections and ground truth for the threshold and gallery tests."""

import numpy as np

from defectlens.eval.detection import Detections, GroundTruth


def random_detections(
    rng: np.random.Generator, n_images: int
) -> tuple[list[Detections], list[GroundTruth]]:
    dets, gts = [], []
    for _ in range(n_images):
        n_gt = int(rng.integers(0, 4))
        xy = rng.uniform(0, 80, size=(n_gt, 2))
        gt_boxes = np.concatenate([xy, xy + rng.uniform(10, 30, size=(n_gt, 2))], axis=1)
        gt_labels = rng.integers(1, 3, size=n_gt)
        jitter = rng.normal(0, 3, size=gt_boxes.shape)
        n_noise = int(rng.integers(0, 5))
        nxy = rng.uniform(0, 80, size=(n_noise, 2))
        noise = np.concatenate([nxy, nxy + rng.uniform(10, 30, size=(n_noise, 2))], axis=1)
        boxes = np.concatenate([gt_boxes + jitter, noise])
        labels = np.concatenate([gt_labels, rng.integers(1, 3, size=n_noise)])
        scores = np.round(rng.uniform(0.01, 1, size=len(boxes)), 2)  # rounded: ties happen
        dets.append(Detections.of(boxes, scores, labels))
        gts.append(GroundTruth.of(gt_boxes, gt_labels))
    return dets, gts
