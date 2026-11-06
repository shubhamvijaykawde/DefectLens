"""Shared test fixtures."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from defectlens.data.coco import to_coco, write_coco
from defectlens.data.neu_det import Box, Record

CLASSES = ("crazing", "inclusion")


@pytest.fixture
def tiny_data(tmp_path: Path) -> tuple[Path, Path]:
    """8 grayscale 100x80 images with one bright rectangle each, as images + COCO splits."""
    images, processed = tmp_path / "raw" / "IMAGES", tmp_path / "processed"
    images.mkdir(parents=True)
    records = []
    for i in range(8):
        img = np.full((80, 100), 60, dtype=np.uint8)
        img[20:50, 30:70] = 230
        name = f"{CLASSES[i % 2]}_{i}"
        cv2.imwrite(str(images / f"{name}.jpg"), img)
        records.append(Record(name, f"{name}.jpg", 100, 80, (Box(CLASSES[i % 2], 30, 20, 70, 50),)))
    coco = to_coco(records)
    for split in ("train", "val", "test"):
        write_coco(coco, processed / f"{split}.json")
    return tmp_path / "raw", processed
