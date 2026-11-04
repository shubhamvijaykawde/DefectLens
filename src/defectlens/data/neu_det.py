"""NEU-DET: Pascal VOC XML parsing and annotation validation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

CLASSES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)


@dataclass(frozen=True)
class Box:
    """Absolute-pixel box, ``xmax``/``ymax`` exclusive-style as stored by VOC (xmin < xmax)."""

    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class Record:
    """One image with its boxes."""

    image_id: str  # file stem, e.g. "crazing_1"
    file_name: str  # relative to the images dir
    width: int
    height: int
    boxes: tuple[Box, ...]


def parse_voc_xml(xml_path: Path) -> Record:
    """Parse one VOC annotation file."""
    root = ET.parse(xml_path).getroot()  # noqa: S314 (trusted, hash-verified dataset)
    size = root.find("size")
    if size is None:
        msg = f"{xml_path}: missing <size>"
        raise ValueError(msg)
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        name = obj.findtext("name")
        if bb is None or name is None:
            msg = f"{xml_path}: object without name/bndbox"
            raise ValueError(msg)
        boxes.append(
            Box(
                name.strip(),
                *(float(bb.findtext(k) or "nan") for k in ("xmin", "ymin", "xmax", "ymax")),
            )
        )
    # The <filename> tag is unreliable (missing the extension for patches_* and pitted_surface_*),
    # so the image is always "<annotation stem>.jpg"; validate_records checks that it exists.
    file_name = f"{xml_path.stem}.jpg"
    return Record(
        image_id=xml_path.stem,
        file_name=file_name,
        width=int(size.findtext("width") or 0),
        height=int(size.findtext("height") or 0),
        boxes=tuple(boxes),
    )


def load_records(raw_dir: Path) -> list[Record]:
    """Load every annotation under ``raw_dir/ANNOTATIONS``, sorted by image id."""
    ann_dir = raw_dir / "ANNOTATIONS"
    return sorted((parse_voc_xml(p) for p in ann_dir.glob("*.xml")), key=lambda r: r.image_id)


def drop_duplicate_boxes(records: list[Record]) -> tuple[list[Record], list[str]]:
    """Remove exact duplicate boxes (same label and coordinates), keeping the first.

    The raw NEU-DET annotations contain a few; a duplicate would count as a false negative
    or inflate recall in evaluation. Returns the cleaned records and one message per drop.
    """
    out: list[Record] = []
    notes: list[str] = []
    for r in records:
        seen: set[Box] = set()
        kept: list[Box] = []
        for b in r.boxes:
            if b in seen:
                notes.append(f"{r.image_id}: dropped duplicate box {b}")
                continue
            seen.add(b)
            kept.append(b)
        out.append(replace(r, boxes=tuple(kept)))
    return out, notes


def image_class(record: Record) -> str:
    """Stratification key: the class in the file name (``crazing_104`` -> ``crazing``).

    123 of 1,800 images carry boxes of more than one class, so the box labels alone cannot give
    one key per image. The file-name class is the dataset's own 300-per-class grouping, and it
    is always one of the image's box labels (checked here).
    """
    prefix = record.image_id.rsplit("_", 1)[0]
    if prefix not in {b.label for b in record.boxes}:
        msg = f"{record.image_id}: file-name class {prefix!r} not among its box labels"
        raise ValueError(msg)
    return prefix


def validate_records(records: list[Record], images_dir: Path | None = None) -> list[str]:
    """Return human-readable problems: unknown classes, empty images, degenerate or
    out-of-bounds boxes, duplicate boxes, duplicate ids, missing image files."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    for r in records:
        if r.image_id in seen_ids:
            problems.append(f"{r.image_id}: duplicate image id")
        seen_ids.add(r.image_id)
        if not r.boxes:
            problems.append(f"{r.image_id}: no boxes")
        if images_dir is not None and not (images_dir / r.file_name).is_file():
            problems.append(f"{r.image_id}: image file {r.file_name} missing")
        seen_boxes: set[tuple[str, float, float, float, float]] = set()
        for b in r.boxes:
            if b.label not in CLASSES:
                problems.append(f"{r.image_id}: unknown class {b.label!r}")
            if not (b.xmax > b.xmin and b.ymax > b.ymin):
                problems.append(f"{r.image_id}: empty/inverted box {b}")
            if b.xmin < 0 or b.ymin < 0 or b.xmax > r.width or b.ymax > r.height:
                problems.append(f"{r.image_id}: box out of bounds {b} for {r.width}x{r.height}")
            key = (b.label, b.xmin, b.ymin, b.xmax, b.ymax)
            if key in seen_boxes:
                problems.append(f"{r.image_id}: duplicate box {b}")
            seen_boxes.add(key)
    return problems
