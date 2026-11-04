from pathlib import Path

import pytest

from defectlens.data.neu_det import (
    Box,
    Record,
    drop_duplicate_boxes,
    image_class,
    parse_voc_xml,
    validate_records,
)

FIX = Path(__file__).parent / "fixtures" / "neu_det"


def test_parse_uses_stem_for_file_name() -> None:
    r = parse_voc_xml(FIX / "patches_2.xml")  # XML <filename> lacks the extension
    assert r.file_name == "patches_2.jpg"
    assert (r.width, r.height) == (200, 100)
    assert r.boxes == (Box("patches", 0, 0, 250, 50),)


def test_duplicates_dropped_and_reported() -> None:
    r = parse_voc_xml(FIX / "crazing_1.xml")
    assert len(r.boxes) == 3
    (clean,), notes = drop_duplicate_boxes([r])
    assert len(clean.boxes) == 2 and len(notes) == 1
    assert validate_records([clean]) == []


def test_validate_flags_out_of_bounds_and_duplicates() -> None:
    r = parse_voc_xml(FIX / "patches_2.xml")
    problems = validate_records([r])
    assert any("out of bounds" in p for p in problems)
    dup = parse_voc_xml(FIX / "crazing_1.xml")
    assert any("duplicate box" in p for p in validate_records([dup]))


def test_validate_flags_empty_inverted_unknown_and_missing_image(tmp_path: Path) -> None:
    bad = Record(
        "crazing_9",
        "crazing_9.jpg",
        10,
        10,
        (Box("crazing", 5, 5, 5, 9), Box("dent", 0, 0, 5, 5)),
    )
    empty = Record("crazing_8", "crazing_8.jpg", 10, 10, ())
    problems = " | ".join(validate_records([bad, empty], tmp_path))
    for needle in ("empty/inverted", "unknown class", "no boxes", "missing"):
        assert needle in problems


def test_image_class_is_file_name_class() -> None:
    r = parse_voc_xml(FIX / "crazing_1.xml")  # also has a patches box
    assert image_class(r) == "crazing"
    with pytest.raises(ValueError, match="not among"):
        image_class(Record("scratches_1", "scratches_1.jpg", 10, 10, (Box("patches", 0, 0, 5, 5),)))
