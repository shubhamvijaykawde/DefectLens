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


@pytest.fixture
def tiny_visa(tmp_path: Path) -> dict[str, Path]:
    """A 2-category-free VisA-like tree for category ``cat``: 12 normal train images and
    5 anomalous + 5 normal images in each of the two test halves, 64x48 colour, with masks."""
    import csv
    import json

    root = tmp_path / "visa"
    rng = np.random.default_rng(0)
    rows: list[dict[str, str]] = []
    splits: dict[str, list[str]] = {
        "train": [],
        "threshold_calibration": [],
        "threshold_evaluation": [],
    }

    def add(split: str, label: str, index: int, key: str) -> None:
        kind = "Anomaly" if label == "anomaly" else "Normal"
        rel = f"cat/Data/Images/{kind}/{key}{index:03d}.JPG"
        mask_rel = f"cat/Data/Masks/{kind}/{key}{index:03d}.png" if label == "anomaly" else ""
        img = (rng.normal(100, 4, (48, 64, 3))).clip(0, 255).astype(np.uint8)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        if label == "anomaly":
            img[10:26, 20:44] = 240
            mask = np.zeros((48, 64), dtype=np.uint8)
            mask[10:26, 20:44] = 255
            (root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / mask_rel), mask)
        cv2.imwrite(str(root / rel), img)
        rows.append(
            {
                "object": "cat",
                "split": "train" if split == "train" else "test",
                "label": label,
                "image": rel,
                "mask": mask_rel,
            }
        )
        splits[split].append(rel)

    for i in range(12):
        add("train", "normal", i, "t")
    for split in ("threshold_calibration", "threshold_evaluation"):
        for i in range(5):
            add(split, "anomaly", i, split[10:13])
            add(split, "normal", i, split[10:13])
    csv_path = root / "1cls.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["object", "split", "label", "image", "mask"])
        writer.writeheader()
        writer.writerows(rows)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "visa_cat.json").write_text(json.dumps({"meta": {"category": "cat"}, **splits}))
    return {"root": root, "csv": csv_path, "splits": splits_dir}


@pytest.fixture
def tiny_typed_visa(tmp_path: Path) -> dict[str, Path]:
    """A VisA-like tree for category ``cat`` with defect types, 96x128 colour images.

    ``train``: 6 normal. Each test half: 6 normal plus melt x3, missing x2, bent x2 and one
    multi-type image (calibration: melt+missing, evaluation: bent+missing); the evaluation half
    has melt x2 instead of x3. Masks are one block per image (two for melt).
    """
    import csv
    import json

    root = tmp_path / "visa"
    rng = np.random.default_rng(1)
    rows: list[dict[str, str]] = []
    anno: list[dict[str, str]] = []
    splits: dict[str, list[str]] = {
        "train": [], "threshold_calibration": [], "threshold_evaluation": [],
    }

    def add(split: str, label: str, key: str, i: int) -> None:
        normal = label == "normal"
        rel = f"cat/Data/Images/{'Normal' if normal else 'Anomaly'}/{key}{i:02d}.JPG"
        mask_rel = "" if normal else f"cat/Data/Masks/Anomaly/{key}{i:02d}.png"
        img = rng.normal(100, 4, (96, 128, 3)).clip(0, 255).astype(np.uint8)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        if not normal:
            mask = np.zeros((96, 128), dtype=np.uint8)
            mask[20:36, 30:50] = 255
            if "melt" in label:
                mask[60:68, 80:90] = 255
            img[mask > 0] = 240
            (root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / mask_rel), mask)
        cv2.imwrite(str(root / rel), img)
        rows.append({"object": "cat", "split": "train" if split == "train" else "test",
                     "label": "normal" if normal else "anomaly", "image": rel, "mask": mask_rel})
        anno.append({"image": rel, "label": label, "mask": mask_rel})
        splits[split].append(rel)

    for i in range(6):
        add("train", "normal", "tr", i)
    layout = {
        "threshold_calibration": ["melt"] * 3 + ["missing"] * 2 + ["bent"] * 2 + ["melt,missing"],
        "threshold_evaluation": ["melt"] * 2 + ["missing"] * 2 + ["bent"] * 2 + ["bent,missing"],
    }
    for split, labels in layout.items():
        for i, label in enumerate(labels):
            add(split, label, f"{split[10:13]}a", i)
        for i in range(6):
            add(split, "normal", f"{split[10:13]}n", i)

    def write(path: Path, fields: list[str], data: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    write(root / "1cls.csv", ["object", "split", "label", "image", "mask"], rows)
    write(root / "cat" / "image_anno.csv", ["image", "label", "mask"], anno)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "visa_cat.json").write_text(json.dumps({"meta": {"category": "cat"}, **splits}))
    return {"root": root, "csv": root / "1cls.csv", "splits": splits_dir}
