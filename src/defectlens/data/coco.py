"""Convert NEU-DET records to COCO detection JSON (boxes as ``[x, y, w, h]``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from defectlens.data.neu_det import CLASSES, Record
from defectlens.eval.detection import Detections


def to_coco(records: list[Record], description: str = "NEU-DET") -> dict[str, Any]:
    """Build a COCO dict. Category ids are 1..6 in ``CLASSES`` order (0 is background);
    image and annotation ids are assigned in sorted ``image_id`` order, so output is stable."""
    cat_id = {name: i + 1 for i, name in enumerate(CLASSES)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for img_id, r in enumerate(sorted(records, key=lambda r: r.image_id), start=1):
        images.append(
            {"id": img_id, "file_name": r.file_name, "width": r.width, "height": r.height}
        )
        for b in r.boxes:
            w, h = b.xmax - b.xmin, b.ymax - b.ymin
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id[b.label],
                    "bbox": [b.xmin, b.ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {
        "info": {"description": description},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cat_id[n], "name": n} for n in CLASSES],
    }


def write_coco(coco: dict[str, Any], path: Path) -> None:
    """Write deterministic JSON (sorted keys, fixed indent, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco, indent=1, sort_keys=True) + "\n")


def subset_coco(coco: dict[str, Any], image_stems: set[str]) -> dict[str, Any]:
    """Keep only images whose file stem is in ``image_stems`` (ids unchanged)."""
    imgs = [i for i in coco["images"] if Path(i["file_name"]).stem in image_stems]
    keep = {i["id"] for i in imgs}
    return {
        **coco,
        "images": imgs,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
    }


@dataclass(frozen=True)
class CocoImage:
    """One image of a COCO split with its boxes as ``xyxy`` pixels and category ids."""

    image_id: int
    file_name: str
    width: int
    height: int
    boxes: npt.NDArray[np.float64]  # (N, 4) xyxy
    labels: npt.NDArray[np.int64]  # (N,) category ids (1..K)


def read_coco(path: Path) -> tuple[list[CocoImage], dict[int, str]]:
    """Read a COCO detection file. Images come in id order; also returns ``{id: name}``."""
    coco = json.loads(path.read_text())
    per_image: dict[int, list[dict[str, Any]]] = {}
    for a in coco["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)
    images = []
    for im in sorted(coco["images"], key=lambda i: i["id"]):
        anns = per_image.get(im["id"], [])
        boxes = np.array(
            [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
             for a in anns],
            dtype=np.float64,
        ).reshape(-1, 4)
        labels = np.array([a["category_id"] for a in anns], dtype=np.int64)
        images.append(
            CocoImage(im["id"], im["file_name"], im["width"], im["height"], boxes, labels)
        )
    return images, {c["id"]: c["name"] for c in coco["categories"]}


def read_predictions(path: Path, images: list[CocoImage]) -> list[Detections]:
    """Read COCO-results JSON (``image_id, category_id, score, bbox=[x, y, w, h]``) as
    per-image detections aligned with ``images`` (images without predictions get none)."""
    by_image: dict[int, list[dict[str, Any]]] = {}
    for r in json.loads(path.read_text()):
        by_image.setdefault(r["image_id"], []).append(r)
    out: list[Detections] = []
    for im in images:
        rs = by_image.get(im.image_id, [])
        boxes = [[r["bbox"][0], r["bbox"][1], r["bbox"][0] + r["bbox"][2],
                  r["bbox"][1] + r["bbox"][3]] for r in rs]
        out.append(Detections.of(boxes, [r["score"] for r in rs], [r["category_id"] for r in rs]))
    return out
