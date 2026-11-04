import json
from pathlib import Path

from pycocotools.coco import COCO

from defectlens.data.coco import subset_coco, to_coco, write_coco
from defectlens.data.neu_det import Box, Record


def _records() -> list[Record]:
    return [
        Record("scratches_2", "scratches_2.jpg", 200, 200, (Box("scratches", 10, 20, 60, 40),)),
        Record(
            "crazing_1",
            "crazing_1.jpg",
            200,
            200,
            (Box("crazing", 0, 0, 100, 50), Box("patches", 5, 5, 10, 15)),
        ),
    ]


def test_to_coco_is_sorted_and_xywh() -> None:
    coco = to_coco(_records())
    assert [i["file_name"] for i in coco["images"]] == ["crazing_1.jpg", "scratches_2.jpg"]
    assert [c["name"] for c in coco["categories"]][0] == "crazing"
    assert coco["categories"][0]["id"] == 1  # 0 is reserved for background
    a = coco["annotations"][-1]
    assert a["bbox"] == [10, 20, 50, 20] and a["area"] == 1000 and a["category_id"] == 6


def test_pycocotools_loads_it(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    write_coco(to_coco(_records()), p)
    c = COCO(str(p))
    assert len(c.getImgIds()) == 2 and len(c.getAnnIds()) == 3


def test_write_is_deterministic_and_subset(tmp_path: Path) -> None:
    coco = to_coco(_records())
    write_coco(coco, tmp_path / "a.json")
    write_coco(json.loads((tmp_path / "a.json").read_text()), tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    sub = subset_coco(coco, {"crazing_1"})
    assert len(sub["images"]) == 1 and len(sub["annotations"]) == 2
