"""Render a before/after grid of the preprocessing pipeline on fixed NEU-DET train images.

Columns: original | default pipeline | default + CLAHE. Boxes are drawn in output coordinates,
so a box that sits on its defect in every column shows the remapping is correct.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from defectlens.data.neu_det import CLASSES, drop_duplicate_boxes, load_records
from defectlens.data.splits import load_splits
from defectlens.preprocess.pipeline import PreprocessConfig, Preprocessor

RAW = Path("data/raw/neu_det/NEU-DET")
OUT = Path("reports/figures/preprocess_grid.png")
CELL = 320
COLOURS = [
    (60, 60, 255),
    (60, 200, 60),
    (255, 120, 40),
    (0, 200, 255),
    (220, 60, 220),
    (200, 200, 0),
]


def draw(img: np.ndarray, boxes: np.ndarray, labels: list[str], title: str) -> np.ndarray:
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    scale = CELL / max(canvas.shape[:2])
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    pad = np.full((CELL, CELL, 3), 114, np.uint8)
    pad[: canvas.shape[0], : canvas.shape[1]] = canvas
    for (x0, y0, x1, y1), lab in zip(boxes * scale, labels, strict=True):
        col = COLOURS[CLASSES.index(lab)]
        cv2.rectangle(pad, (round(x0), round(y0)), (round(x1), round(y1)), col, 2)
    cv2.putText(pad, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return pad


def main() -> int:
    cfg = PreprocessConfig.from_dict(
        OmegaConf.to_container(OmegaConf.load("configs/preprocess/default.yaml"), resolve=True)
    )
    splits = load_splits(Path("data/splits/neu_det.json"))
    records = {r.image_id: r for r in drop_duplicate_boxes(load_records(RAW))[0]}
    # first train image of every class (sorted ids), so the figure never changes
    picks = [next(i for i in splits["train"] if i.rsplit("_", 1)[0] == c) for c in CLASSES]
    plain, clahe = Preprocessor(cfg), Preprocessor(replace(cfg, clahe=True))
    rows = []
    for image_id in picks:
        r = records[image_id]
        img = cv2.imread(str(RAW / "IMAGES" / r.file_name), cv2.IMREAD_GRAYSCALE)
        assert img is not None, r.file_name
        boxes = np.array([[b.xmin, b.ymin, b.xmax, b.ymax] for b in r.boxes])
        labels = [b.label for b in r.boxes]
        cols = [draw(img, boxes, labels, f"{image_id} original {img.shape[1]}x{img.shape[0]}")]
        for name, pre in (("default", plain), ("+CLAHE", clahe)):
            res = pre(img, boxes)
            cols.append(draw(res.image, res.boxes, labels, f"{name} {res.image.shape[1]}px"))
        rows.append(np.hstack(cols))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), np.vstack(rows))
    print(f"wrote {OUT} ({len(rows)} rows x 3 columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
