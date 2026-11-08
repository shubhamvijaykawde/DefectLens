"""VisA defect masks -> detection boxes and defect-type labels (Phase 4b).

VisA ships one binary mask per anomalous image and, per category, an ``image_anno.csv`` whose
``label`` column names the defect type(s): ``melt``, or ``melt,scratch`` for an image with two.
A mask does not say which pixels belong to which type, so only single-type images get typed
boxes; ``build_visa_coco`` refuses a multi-type image rather than guess.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from defectlens.data.visa_splits import read_one_cls

NORMAL = "normal"
Types = tuple[str, ...]


@dataclass(frozen=True)
class BoxConfig:
    """How mask pixels become boxes (pixels of the original image).

    Defect masks are often several fragments of one defect (a melted joint is up to 25 specks),
    so fragments closer than ``merge_px`` form one box. Boxes with fewer than ``min_area`` mask
    pixels are annotation noise and are dropped; the rest grow by ``pad_px`` on every side.
    """

    merge_px: int = 25
    min_area: int = 20
    pad_px: int = 6


DEFAULT_BOX = BoxConfig()


def parse_types(label: str) -> Types:
    """``"melt,scratch"`` -> ``("melt", "scratch")`` (sorted); ``"normal"`` -> ``()``."""
    if label == NORMAL:
        return ()
    types = tuple(sorted({t.strip() for t in label.split(",") if t.strip()}))
    if not types:
        msg = f"empty defect label {label!r}"
        raise ValueError(msg)
    return types


def read_type_labels(root: Path, csv_path: Path, category: str) -> dict[str, Types]:
    """Defect types of every image of ``category`` in the official ``1cls.csv``, keyed by the
    image path used everywhere else. Normal images map to ``()``.

    The types come from ``<root>/<category>/image_anno.csv``. An image that is anomalous in one
    file and normal in the other is a data error, so it raises.
    """
    with (root / category / "image_anno.csv").open() as f:
        anno = {r["image"]: parse_types(r["label"]) for r in csv.DictReader(f)}
    out: dict[str, Types] = {}
    for row in read_one_cls(csv_path, category):
        path = row["image"]
        if path not in anno:
            msg = f"{path} is in {csv_path} but not in {category}/image_anno.csv"
            raise ValueError(msg)
        if (row["label"] == "anomaly") != bool(anno[path]):
            msg = f"{path}: label {row['label']!r} in the split file disagrees with the types"
            raise ValueError(msg)
        out[path] = anno[path]
    return out


def mask_to_boxes(
    mask: npt.NDArray[np.generic], cfg: BoxConfig = DEFAULT_BOX
) -> npt.NDArray[np.float64]:
    """Boxes ``(N, 4)`` as ``xyxy`` for the foreground of a binary mask (any non-zero pixel).

    Foreground fragments within ``cfg.merge_px`` of each other are merged (by dilating, labelling
    and boxing the *original* pixels of each merged region), then boxes below ``cfg.min_area``
    are dropped and the rest are padded and clipped to the image.
    """
    binary: npt.NDArray[np.uint8] = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = binary.shape
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    merged = binary
    if cfg.merge_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.merge_px, cfg.merge_px))
        merged = np.asarray(cv2.dilate(binary, kernel), dtype=np.uint8)
    _, labels = cv2.connectedComponents(merged, connectivity=8)
    region = labels[ys, xs]
    boxes: list[list[float]] = []
    for r in np.unique(region):
        sel = region == r
        if int(sel.sum()) < cfg.min_area:
            continue
        x0, x1 = int(xs[sel].min()), int(xs[sel].max()) + 1
        y0, y1 = int(ys[sel].min()), int(ys[sel].max()) + 1
        boxes.append([
            max(0, x0 - cfg.pad_px), max(0, y0 - cfg.pad_px),
            min(w, x1 + cfg.pad_px), min(h, y1 + cfg.pad_px),
        ])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def build_visa_coco(
    root: Path,
    csv_path: Path,
    category: str,
    paths: Sequence[str],
    classes: Sequence[str],
    type_labels: Mapping[str, Types] | None,
    box_cfg: BoxConfig = DEFAULT_BOX,
) -> dict[str, Any]:
    """COCO detection dict for ``paths`` (file names relative to ``root``; ids follow sorted paths).

    Category ids are 1..K in ``classes`` order. With ``type_labels=None`` the images carry no
    boxes (for scoring). Otherwise normal images get none, and an anomalous image must have exactly
    one type, that type must be in ``classes`` and the mask must give at least one box; anything
    else raises, because a silently unlabelled defect would train the detector to ignore it.
    """
    masks = {r["image"]: r["mask"] for r in read_one_cls(csv_path, category)}
    cat_id = {name: i + 1 for i, name in enumerate(classes)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for image_id, path in enumerate(sorted(paths), start=1):
        img = cv2.imread(str(root / path), cv2.IMREAD_COLOR)
        if img is None:
            msg = f"cannot read image {root / path}"
            raise FileNotFoundError(msg)
        h, w = img.shape[:2]
        images.append({"id": image_id, "file_name": path, "width": w, "height": h})
        types = () if type_labels is None else type_labels[path]
        if not types:
            continue
        if len(types) > 1:
            msg = f"{path} has several defect types {types}: the mask cannot be split by type"
            raise ValueError(msg)
        if types[0] not in cat_id:
            msg = f"{path}: defect type {types[0]!r} is not one of the classes {list(classes)}"
            raise ValueError(msg)
        mask = cv2.imread(str(root / masks[path]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            msg = f"cannot read mask {root / masks[path]}"
            raise FileNotFoundError(msg)
        boxes = mask_to_boxes(mask, box_cfg)
        if len(boxes) == 0:
            msg = f"{path}: the mask gives no box with box_cfg {box_cfg}"
            raise ValueError(msg)
        for x0, y0, x1, y1 in boxes:
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": cat_id[types[0]],
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "area": float((x1 - x0) * (y1 - y0)), "iscrowd": 0,
            })
    return {
        "info": {"description": f"VisA {category}"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cat_id[n], "name": n} for n in classes],
    }
