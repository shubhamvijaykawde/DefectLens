"""VisA image sets from the fixed split files."""

from pathlib import Path

import numpy as np
import pytest
import torch

from defectlens.anomaly.datamodule import (
    VisaImages,
    denormalize,
    few_shot_subset,
    load_split_paths,
)


def test_load_split_paths_and_unknown_split(tiny_visa: dict[str, Path]) -> None:
    assert len(load_split_paths(tiny_visa["splits"], "cat", "train")) == 12
    assert len(load_split_paths(tiny_visa["splits"], "cat", "threshold_calibration")) == 10
    with pytest.raises(ValueError, match="unknown split"):
        load_split_paths(tiny_visa["splits"], "cat", "test")


def test_few_shot_subset_is_fixed_sorted_and_clipped() -> None:
    paths = [f"p{i:02d}" for i in range(20)]
    a, b = few_shot_subset(paths, 5, seed=1), few_shot_subset(paths[::-1], 5, seed=1)
    assert a == b and len(a) == 5 and a == sorted(a)
    assert few_shot_subset(paths, 5, seed=2) != a
    assert few_shot_subset(paths, None, seed=1) == paths
    assert few_shot_subset(paths, 50, seed=1) == paths


def test_visa_images_shapes_labels_masks(tiny_visa: dict[str, Path]) -> None:
    paths = load_split_paths(tiny_visa["splits"], "cat", "threshold_calibration")
    ds = VisaImages(tiny_visa["root"], tiny_visa["csv"], "cat", paths, (32, 40))
    assert len(ds) == 10 and sum(ds.labels) == 5
    for i in range(len(ds)):
        item = ds[i]
        assert item["image"].shape == (3, 32, 40) and item["mask"].shape == (32, 40)
        assert bool(item["mask"].any()) == bool(item["label"])
    anomalous = next(ds[i] for i in range(len(ds)) if ds.labels[i])
    assert float(anomalous["mask"].mean()) == pytest.approx(0.125, abs=0.03)  # 16x24 in 48x64


def test_visa_images_rejects_unknown_path(tiny_visa: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="not in"):
        VisaImages(tiny_visa["root"], tiny_visa["csv"], "cat", ["cat/nope.JPG"], (32, 40))


def test_denormalize_inverts_normalisation(tiny_visa: dict[str, Path]) -> None:
    paths = load_split_paths(tiny_visa["splits"], "cat", "train")[:2]
    ds = VisaImages(tiny_visa["root"], tiny_visa["csv"], "cat", paths, (32, 40))
    batch = torch.stack([ds[0]["image"], ds[1]["image"]])
    img = denormalize(batch)
    assert img.shape == (2, 32, 40, 3) and img.min() >= 0.0 and img.max() <= 1.0
    assert np.isclose(img.mean(), 100 / 255, atol=0.05)
