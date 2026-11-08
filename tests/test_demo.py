"""The Gradio demo (Phase 6b): the function behind the button, the UI it builds, the mount."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from defectlens.serve.app import create_app
from defectlens.serve.config import ServeSettings
from defectlens.serve.demo import (
    NOTE,
    DemoError,
    build_demo,
    create_app_with_demo,
    decode_overlay,
    draw_defects,
    inspect_for_demo,
    mount_demo,
    summarize,
)
from defectlens.serve.inference import Inspector
from defectlens.serve.model_registry import ModelRegistry
from defectlens.serve.schemas import Defect
from serve_fixtures import make_settings, square_png, steel_jpeg


@pytest.fixture
def settings(tmp_path: Path) -> ServeSettings:
    return make_settings(tmp_path)


@pytest.fixture
def inspector(settings: ServeSettings) -> Inspector:
    return Inspector(settings, ModelRegistry(settings))


def write(tmp_path: Path, data: bytes, name: str) -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


# ---- drawing ---------------------------------------------------------------------------------


def test_draw_defects_marks_the_box_and_leaves_the_input_alone() -> None:
    bgr = np.full((100, 100, 3), 30, np.uint8)
    before = bgr.copy()
    out = draw_defects(bgr, [Defect(label="scratches", score=0.9, box=(20, 40, 80, 90))])
    assert out.shape == (100, 100, 3) and np.array_equal(bgr, before)
    assert not np.array_equal(out[40, 20:80], np.full((60, 3), 30, np.uint8))  # top edge drawn
    assert np.array_equal(out[60, 50], [30, 30, 30])  # interior untouched
    assert np.array_equal(draw_defects(bgr, []), bgr[..., ::-1])  # no defect: only BGR -> RGB


def test_draw_defects_gives_each_class_its_own_colour() -> None:
    bgr = np.zeros((200, 200, 3), np.uint8)
    defects = [Defect(label="a", score=0.5, box=(10, 100, 60, 150)),
               Defect(label="b", score=0.5, box=(100, 100, 150, 150))]
    out = draw_defects(bgr, defects)
    assert not np.array_equal(out[125, 10], out[125, 100])  # left edge of each box


def test_decode_overlay_round_trips_a_png_and_rejects_garbage() -> None:
    rgb = decode_overlay(base64.b64encode(square_png()).decode())
    assert rgb.shape == (64, 96, 3) and rgb[30, 40].tolist() == [255, 255, 255]
    with pytest.raises(ValueError, match="PNG"):
        decode_overlay(base64.b64encode(b"nope").decode())


# ---- the button's function -------------------------------------------------------------------


def test_steel_answer_has_boxes_rows_summary_and_the_same_json_as_the_api(
    inspector: Inspector, tmp_path: Path,
) -> None:
    path = write(tmp_path, steel_jpeg(), "a.jpg")
    picture, md, rows, raw = inspect_for_demo(inspector, "steel", path, "10", True)
    api = inspector.inspect("steel", steel_jpeg(), fn_fp_ratio=10.0).model_dump()
    assert picture.shape == (200, 200, 3)
    assert [r[0] for r in rows] == ["inclusion", "scratches"] and rows[0][2:] == [20, 40, 100, 80]
    assert "2 defects" in md and "0.4" in md and "tiny_tv" in md and NOTE in md
    assert raw["defects"] == api["defects"] and raw["operating_point"] == api["operating_point"]
    assert raw["flagged"] is True and raw["anomaly"] is None


def test_anomaly_answer_shows_the_overlay_and_not_the_base64_in_the_json(
    inspector: Inspector, tmp_path: Path,
) -> None:
    path = write(tmp_path, square_png(), "a.png")
    picture, md, rows, raw = inspect_for_demo(inspector, "visa_a", path, "10", True)
    assert picture.shape[2] == 3 and picture.dtype == np.uint8
    assert "ANOMALY" in md and "flagged when score >= threshold" in md
    assert rows and rows[0][0] == "anomaly"  # white block -> a region above the threshold
    assert raw["anomaly"]["flagged"] is True and "heatmap_png_base64" not in raw["anomaly"]


def test_a_normal_anomaly_image_is_not_flagged_and_without_heatmap_falls_back_to_the_image(
    inspector: Inspector, tmp_path: Path,
) -> None:
    black = write(tmp_path, steel_jpeg(), "g.jpg")  # uniform grey: score below the threshold 3.0
    picture, md, rows, raw = inspect_for_demo(inspector, "visa_a", black, "5", False)
    assert "no anomaly flagged" in md and rows == [] and raw["flagged"] is False
    assert picture.shape == (200, 200, 3)  # no overlay requested: the uploaded image


def test_ratio_none_uses_the_configured_default(inspector: Inspector, tmp_path: Path) -> None:
    path = write(tmp_path, steel_jpeg(), "a.jpg")
    _, _, _, raw = inspect_for_demo(inspector, "steel", path, None)
    assert raw["operating_point"]["fn_fp_ratio"] == 10


@pytest.mark.parametrize(
    ("product", "use_image", "ratio", "message"),
    [
        (None, True, "10", "Pick a product"),
        ("steel", False, "10", "Upload an image"),
        ("nope", True, "10", "Unknown product"),
        ("steel", True, "7", "no frozen threshold"),
        ("visa_missing", True, "10", "not available"),
    ],
)
def test_errors_are_readable_messages_not_http_codes(
    inspector: Inspector, tmp_path: Path, product: str | None, use_image: bool, ratio: str,
    message: str,
) -> None:
    path = write(tmp_path, steel_jpeg(), "a.jpg") if use_image else None
    with pytest.raises(DemoError, match=message):
        inspect_for_demo(inspector, product, path, ratio)


def test_a_file_that_is_not_an_image_is_reported(inspector: Inspector, tmp_path: Path) -> None:
    with pytest.raises(DemoError, match="Not a usable image"):
        inspect_for_demo(inspector, "steel", write(tmp_path, b"hello", "x.jpg"), "10")


def test_summary_mentions_the_size_warning(inspector: Inspector) -> None:
    resp = inspector.inspect("steel", square_png(), fn_fp_ratio=10.0)  # 96x64, trained on 200x200
    assert "warning: image is 96x64" in summarize(resp)


# ---- UI and mount ----------------------------------------------------------------------------


def test_build_demo_offers_only_available_products_and_the_frozen_ratios(
    inspector: Inspector,
) -> None:
    demo = build_demo(inspector)
    dropdown = next(b for b in demo.blocks.values() if getattr(b, "label", "") == "Product")
    radio = next(b for b in demo.blocks.values() if getattr(b, "choices", None) is not None
                 and "cost of a miss" in str(getattr(b, "label", "")))
    assert "visa_missing" not in [c[1] for c in dropdown.choices] and len(dropdown.choices) == 5
    assert [c[1] for c in radio.choices] == ["5", "10", "50"] and radio.value == "10"


def test_the_demo_is_mounted_beside_the_api(settings: ServeSettings) -> None:
    with TestClient(create_app_with_demo(settings)) as c:
        page = c.get("/demo/")
        assert page.status_code == 200 and "gradio" in page.text.lower()
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/inspect", params={"product": "steel"},
                   files={"image": ("a.jpg", steel_jpeg(), "image/jpeg")})
        assert r.status_code == 200


def test_mount_demo_reuses_the_apps_inspector(settings: ServeSettings) -> None:
    app = create_app(settings)
    assert mount_demo(app) is app and app.state.inspector.registry is app.state.registry
