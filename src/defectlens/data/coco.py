"""Convert NEU-DET records to COCO detection JSON (boxes as ``[x, y, w, h]``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from defectlens.data.neu_det import CLASSES, Record


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
