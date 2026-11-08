"""VisA masks -> boxes and defect-type labels."""

import csv
from pathlib import Path

import numpy as np
import pytest

from defectlens.data.coco import read_coco, write_coco
from defectlens.data.visa_to_coco import (
    BoxConfig,
    build_visa_coco,
    mask_to_boxes,
    parse_types,
    read_type_labels,
)

CFG = BoxConfig(merge_px=9, min_area=4, pad_px=2)


def _mask(*blocks: tuple[int, int, int, int], shape: tuple[int, int] = (60, 80)) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    for y0, y1, x0, x1 in blocks:
        m[y0:y1, x0:x1] = 255
    return m


def test_parse_types() -> None:
    assert parse_types("normal") == ()
    assert parse_types("melt") == ("melt",)
    assert parse_types("scratch,melt") == ("melt", "scratch")
    with pytest.raises(ValueError, match="empty"):
        parse_types("")


def test_mask_to_boxes_merges_close_fragments_and_keeps_far_ones_apart() -> None:
    # two 4x4 fragments 4 px apart merge; a third one 30 px away stays separate
    boxes = mask_to_boxes(_mask((10, 14, 10, 14), (10, 14, 18, 22), (40, 44, 60, 64)), CFG)
    assert boxes.shape == (2, 4)
    got = sorted(boxes.tolist())
    assert got[0] == [8.0, 8.0, 24.0, 16.0]  # union of the two fragments, padded by 2
    assert got[1] == [58.0, 38.0, 66.0, 46.0]


def test_mask_to_boxes_drops_specks_and_clips_to_the_image() -> None:
    m = _mask((0, 5, 0, 5))
    m[30, 40] = 255  # a one-pixel speck: below min_area
    boxes = mask_to_boxes(m, CFG)
    assert boxes.tolist() == [[0.0, 0.0, 7.0, 7.0]]  # padding cannot go below 0
    assert mask_to_boxes(np.zeros((10, 10), dtype=np.uint8), CFG).shape == (0, 4)
    edge = mask_to_boxes(_mask((55, 60, 75, 80)), CFG)
    assert edge.tolist() == [[73.0, 53.0, 80.0, 60.0]]  # clipped to width 80, height 60


def test_mask_to_boxes_without_merging_keeps_fragments() -> None:
    boxes = mask_to_boxes(_mask((10, 14, 10, 14), (10, 14, 18, 22)), BoxConfig(1, 4, 0))
    assert len(boxes) == 2


def test_read_type_labels(tiny_typed_visa: dict[str, Path]) -> None:
    types = read_type_labels(tiny_typed_visa["root"], tiny_typed_visa["csv"], "cat")
    assert len(types) == 6 + (8 + 6) + (7 + 6)  # train, calibration, evaluation
    assert sum(1 for t in types.values() if not t) == 6 + 12
    assert sorted(t for t in types.values() if len(t) == 2) == [("bent", "missing"),
                                                                ("melt", "missing")]


def test_read_type_labels_rejects_disagreeing_files(tiny_typed_visa: dict[str, Path]) -> None:
    anno = tiny_typed_visa["root"] / "cat" / "image_anno.csv"
    rows = list(csv.DictReader(anno.open()))
    rows[-1]["label"] = "melt"  # a normal image that the type file calls anomalous
    with anno.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "label", "mask"])
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="disagrees"):
        read_type_labels(tiny_typed_visa["root"], tiny_typed_visa["csv"], "cat")


def test_build_visa_coco_boxes_classes_and_round_trip(
    tiny_typed_visa: dict[str, Path], tmp_path: Path
) -> None:
    root, csv_path = tiny_typed_visa["root"], tiny_typed_visa["csv"]
    types = read_type_labels(root, csv_path, "cat")
    single = sorted(p for p, t in types.items() if len(t) <= 1)
    classes = ["bent", "melt", "missing"]
    coco = build_visa_coco(root, csv_path, "cat", single, classes, types, CFG)
    assert [c["name"] for c in coco["categories"]] == classes
    assert [i["file_name"] for i in coco["images"]] == single  # ids follow sorted paths
    assert coco["images"][0]["width"] == 128 and coco["images"][0]["height"] == 96
    by_image: dict[int, list[dict[str, object]]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    for img in coco["images"]:
        t = types[img["file_name"]]
        anns = by_image.get(img["id"], [])
        if not t:
            assert anns == []  # normal images are negatives
        elif t == ("melt",):  # two blocks 20 px apart: not merged at merge_px=9
            assert len(anns) == 2 and {a["category_id"] for a in anns} == {2}
        else:
            assert len(anns) == 1
            assert anns[0]["category_id"] == classes.index(t[0]) + 1
            assert anns[0]["bbox"] == [28.0, 18.0, 24.0, 20.0]  # block (20:36, 30:50) + pad 2
    write_coco(coco, tmp_path / "c.json")
    images, names = read_coco(tmp_path / "c.json")
    assert len(images) == len(single) and names == {1: "bent", 2: "melt", 3: "missing"}


def test_build_visa_coco_without_types_has_no_boxes(tiny_typed_visa: dict[str, Path]) -> None:
    root, csv_path = tiny_typed_visa["root"], tiny_typed_visa["csv"]
    types = read_type_labels(root, csv_path, "cat")
    coco = build_visa_coco(root, csv_path, "cat", sorted(types)[:5], ["melt"], None, CFG)
    assert len(coco["images"]) == 5 and coco["annotations"] == []


def test_build_visa_coco_refuses_ambiguous_or_unknown_types(
    tiny_typed_visa: dict[str, Path],
) -> None:
    root, csv_path = tiny_typed_visa["root"], tiny_typed_visa["csv"]
    types = read_type_labels(root, csv_path, "cat")
    multi = next(p for p, t in types.items() if len(t) == 2)
    with pytest.raises(ValueError, match="several defect types"):
        build_visa_coco(root, csv_path, "cat", [multi], ["bent", "melt", "missing"], types, CFG)
    melt = next(p for p, t in types.items() if t == ("melt",))
    with pytest.raises(ValueError, match="not one of the classes"):
        build_visa_coco(root, csv_path, "cat", [melt], ["bent", "missing"], types, CFG)
    with pytest.raises(ValueError, match="no box"):  # a mask that is all specks
        build_visa_coco(root, csv_path, "cat", [melt], ["melt"], types,
                        BoxConfig(merge_px=9, min_area=10_000, pad_px=0))
