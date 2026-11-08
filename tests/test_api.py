"""The HTTP API (Phase 6a) on CPU with tiny hand-built ONNX models (see ``serve_fixtures``)."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from defectlens.eval.thresholds import FrozenEntry, ProtocolError, write_frozen
from defectlens.serve import model_registry as registry_module
from defectlens.serve.app import create_app
from defectlens.serve.config import ServeSettings, load_settings
from defectlens.serve.model_registry import (
    CPU,
    CUDA,
    ModelRegistry,
    ModelUnavailable,
    create_session,
)
from serve_fixtures import make_settings, square_png, steel_jpeg

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> ServeSettings:
    return make_settings(tmp_path)


@pytest.fixture
def client(settings: ServeSettings) -> TestClient:
    with TestClient(create_app(settings)) as c:
        return c


def post(client: TestClient, product: str, data: bytes, name: str = "x.png", **params: object):  # type: ignore[no-untyped-def]
    return client.post("/inspect", params={"product": product, **params},
                       files={"image": (name, data)})


# ---- /health and /products ------------------------------------------------------------------


def test_health_reports_provider_and_what_is_loaded(client: TestClient) -> None:
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["provider"] == CPU and h["fallback_reason"] == ""
    assert h["loaded"] == [] and h["vram_in_use_mib"] == 0
    assert "visa_missing" in h["unavailable_products"]
    assert "not found" in h["unavailable_products"]["visa_missing"]
    assert set(h["available_products"]) == {"steel", "steel_dfine", "visa_a", "visa_b", "visa_c"}
    post(client, "steel", steel_jpeg(), "a.jpg")
    h = client.get("/health").json()
    assert h["loaded"] == [{"product": "steel", "provider": CPU, "tf32": False, "vram_mib": 100}]


def test_cuda_requested_but_missing_falls_back_to_cpu_and_says_why(settings: ServeSettings) -> None:
    s = settings.model_copy(update={"providers": [CUDA, CPU]})
    reg = ModelRegistry(s, available_providers=[CPU])
    h = TestClient(create_app(s, registry=reg)).get("/health").json()
    assert h["provider"] == CPU and CUDA in h["fallback_reason"]
    assert h["requested_providers"] == [CUDA, CPU]


def test_cuda_in_the_build_but_not_loadable_is_reported_not_assumed(
    settings: ServeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORT lists CUDA as available when its library exists, even if cuDNN / cuBLAS are not found
    (seen in the first live run, where the session silently ran on CPU)."""
    monkeypatch.setattr(registry_module, "probe_cuda", lambda: (False, "libcublasLt.so.13 missing"))
    s = settings.model_copy(update={"providers": [CUDA, CPU]})
    reg = ModelRegistry(s, available_providers=[CUDA, CPU])
    h = TestClient(create_app(s, registry=reg)).get("/health").json()
    assert h["provider"] == CPU and h["fallback_reason"] == "libcublasLt.so.13 missing"
    with pytest.raises(ModelUnavailable, match="libcublasLt"):
        ModelRegistry(s.model_copy(update={"require_cuda": True}), available_providers=[CUDA, CPU])


def test_probe_cuda_is_a_bool_and_a_reason() -> None:
    ok, why = registry_module.probe_cuda()
    assert isinstance(ok, bool) and (ok or why)


def test_require_cuda_refuses_to_start_without_it(settings: ServeSettings) -> None:
    s = settings.model_copy(update={"providers": [CUDA, CPU], "require_cuda": True})
    with pytest.raises(ModelUnavailable, match="require_cuda"):
        ModelRegistry(s, available_providers=[CPU])


def test_products_lists_thresholds_and_availability(client: TestClient) -> None:
    by_name = {p["name"]: p for p in client.get("/products").json()}
    assert by_name["steel"]["thresholds"] == {"5": 0.1, "10": 0.4, "50": 0.5}
    assert by_name["steel"]["available"] and not by_name["visa_missing"]["available"]
    assert by_name["visa_a"]["kind"] == "anomaly" and by_name["visa_a"]["default_fn_fp_ratio"] == 10


# ---- steel -----------------------------------------------------------------------------------


def test_torchvision_detector_returns_boxes_in_image_pixels_above_threshold(
    client: TestClient,
) -> None:
    r = post(client, "steel", steel_jpeg(), "a.jpg")  # default ratio 10 -> threshold 0.4
    assert r.status_code == 200
    j = r.json()
    assert j["product"] == "steel" and j["model"] == "tiny_tv" and j["image_size"] == [200, 200]
    assert j["flagged"] is True and j["anomaly"] is None and j["warnings"] == []
    assert [d["label"] for d in j["defects"]] == ["inclusion", "scratches"]
    assert [round(d["score"], 2) for d in j["defects"]] == [0.9, 0.5]
    assert j["defects"][0]["box"] == pytest.approx([20, 40, 100, 80], abs=1e-3)
    assert j["defects"][1]["box"] == pytest.approx([0, 0, 200, 200], abs=1e-3)
    op = j["operating_point"]
    assert op["threshold"] == 0.4 and op["fn_fp_ratio"] == 10
    assert len(op["frozen_file_sha256"]) == 64
    assert j["provider"] == CPU and j["tf32"] is False
    assert j["timing"]["total_ms"] >= j["timing"]["inference_ms"] > 0


def test_ratio_selects_the_frozen_threshold_and_equal_scores_are_kept(client: TestClient) -> None:
    loose = post(client, "steel", steel_jpeg(), fn_fp_ratio=5).json()
    assert len(loose["defects"]) == 3 and loose["operating_point"]["threshold"] == 0.1
    equal = post(client, "steel", steel_jpeg(), fn_fp_ratio=50).json()  # threshold == score 0.5
    assert [round(d["score"], 2) for d in equal["defects"]] == [0.9, 0.5]


def test_dfine_detector_decodes_logits_and_boxes(client: TestClient) -> None:
    j = post(client, "steel_dfine", steel_jpeg(), "a.jpg").json()
    assert [d["label"] for d in j["defects"]] == ["inclusion", "rolled-in_scale"]
    assert [round(d["score"], 2) for d in j["defects"]] == [0.88, 0.5]
    assert j["defects"][0]["box"] == pytest.approx([30, 30, 70, 70], abs=1e-2)
    assert j["defects"][1]["box"] == pytest.approx([50, 50, 150, 150], abs=1e-2)


def test_unflagged_steel_when_nothing_reaches_the_threshold(
    tmp_path: Path, settings: ServeSettings
) -> None:
    # the threshold of ratio 50 is raised above every score by a fresh frozen file
    from serve_fixtures import frozen_file  # noqa: PLC0415

    entries = [FrozenEntry("neu_det", "tiny_tv", "all", 10.0, 0.95, "val", 1, 0, 0, 0, None, 0.0)]
    write_frozen(tmp_path / "high.json", entries, {"iou": 0.5})
    frozen_file(tmp_path / "unused.json")
    s = settings.model_copy(update={"frozen_thresholds": tmp_path / "high.json"})
    j = post(TestClient(create_app(s)), "steel", steel_jpeg()).json()
    assert j["flagged"] is False and j["defects"] == []


def test_steel_image_of_another_size_gets_a_warning_and_boxes_in_its_own_pixels(
    client: TestClient,
) -> None:
    j = post(client, "steel", steel_jpeg(400), "big.jpg").json()
    assert j["image_size"] == [400, 400] and "400x400" in j["warnings"][0]
    assert j["defects"][0]["box"] == pytest.approx([40, 80, 200, 160], abs=1e-3)


def test_color_upload_is_read_as_grayscale_for_steel(client: TestClient) -> None:
    assert post(client, "steel", square_png(200, 200)).status_code == 200


# ---- anomaly products ------------------------------------------------------------------------


def test_anomaly_product_flags_scores_regions_and_heatmap(client: TestClient) -> None:
    j = post(client, "visa_a", square_png()).json()
    assert j["flagged"] is True and j["image_size"] == [64, 96]
    a = j["anomaly"]
    assert a["score"] == pytest.approx(2.439, abs=0.01) and a["threshold"] == 0.5
    assert a["flagged"] is True and a["map_size"] == [32, 48]
    assert len(j["defects"]) == 1 and j["defects"][0]["label"] == "anomaly"
    assert j["defects"][0]["box"] == pytest.approx([24, 16, 72, 48])
    png = base64.b64decode(a["heatmap_png_base64"], validate=True)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (64, 96, 3)
    assert img[32, 48, 2] > img[2, 2, 2]   # the block is redder than the background


def test_clean_image_is_not_flagged_but_still_gets_a_heatmap(client: TestClient) -> None:
    ok, buf = cv2.imencode(".png", np.zeros((64, 96, 3), np.uint8))
    j = post(client, "visa_a", bytes(buf.tobytes())).json()
    assert j["flagged"] is False and j["defects"] == [] and j["anomaly"]["flagged"] is False
    assert j["anomaly"]["heatmap_png_base64"]


def test_heatmap_can_be_switched_off_and_ratio_changes_the_threshold(client: TestClient) -> None:
    j = post(client, "visa_a", square_png(), heatmap="false", fn_fp_ratio=5).json()
    assert j["anomaly"]["heatmap_png_base64"] is None
    assert j["anomaly"]["threshold"] == 3.0 and j["flagged"] is False  # score 2.44 < 3.0


# ---- errors ----------------------------------------------------------------------------------


@pytest.mark.parametrize("data", [b"", b"not an image", b"%PDF-1.4 ...", b"\x89PNG"])
def test_bad_uploads_are_400(client: TestClient, data: bytes) -> None:
    r = post(client, "steel", data)
    assert r.status_code == 400 and r.json()["detail"]


def test_decompression_bomb_is_400_without_decoding(client: TestClient) -> None:
    import struct  # noqa: PLC0415

    bomb = bytearray(square_png())
    bomb[16:24] = struct.pack(">II", 60000, 60000)
    r = post(client, "visa_a", bytes(bomb))
    assert r.status_code == 400 and "limit" in r.json()["detail"]


def test_unknown_product_is_404_and_lists_the_known_ones(client: TestClient) -> None:
    r = post(client, "gearbox", square_png())
    assert r.status_code == 404 and "steel" in r.json()["detail"]


def test_unknown_ratio_is_422_and_lists_the_frozen_ones(client: TestClient) -> None:
    r = post(client, "steel", steel_jpeg(), fn_fp_ratio=7)
    assert r.status_code == 422 and "available" in r.json()["detail"]


def test_missing_parameters_are_422(client: TestClient) -> None:
    assert client.post("/inspect", params={"product": "steel"}).status_code == 422
    assert client.post("/inspect", files={"image": ("a.png", square_png())}).status_code == 422


def test_oversized_upload_is_413(settings: ServeSettings) -> None:
    s = settings.model_copy(update={"max_upload_bytes": 100})
    r = post(TestClient(create_app(s)), "steel", steel_jpeg())
    assert r.status_code == 413


def test_missing_model_file_is_503_and_other_products_still_work(client: TestClient) -> None:
    r = post(client, "visa_missing", square_png())
    assert r.status_code == 503 and "not found" in r.json()["detail"]
    assert post(client, "visa_a", square_png()).status_code == 200


def test_unloadable_model_is_503_without_internal_details(
    tmp_path: Path, settings: ServeSettings
) -> None:
    bad = tmp_path / "models" / "broken.onnx"
    bad.write_bytes(b"this is not onnx")
    spec = settings.products["visa_a"].model_copy(update={"onnx": bad})
    s = settings.model_copy(update={"products": {**settings.products, "visa_a": spec}})
    r = post(TestClient(create_app(s)), "visa_a", square_png())
    assert r.status_code == 503 and "could not load" in r.json()["detail"]


def test_unexpected_errors_are_500_with_a_generic_message(
    settings: ServeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(settings)

    def boom(*_: object, **__: object) -> None:
        msg = "secret path /srv/x"
        raise RuntimeError(msg)

    monkeypatch.setattr(app.state.inspector, "inspect", boom)
    r = post(TestClient(app, raise_server_exceptions=False), "steel", steel_jpeg())
    assert r.status_code == 500 and "secret" not in r.text


# ---- registry: lazy loading, budget, thresholds ---------------------------------------------


def test_sessions_load_lazily_and_the_least_recently_used_is_dropped(
    settings: ServeSettings,
) -> None:
    reg = ModelRegistry(settings)  # budget 250, each session declares 100
    assert reg.loaded() == []
    reg.acquire("visa_a")
    reg.acquire("visa_b")
    reg.acquire("visa_a")             # a is now the most recent
    reg.acquire("visa_c")             # 3 x 100 > 250: drops b
    assert [m.product for m in reg.loaded()] == ["visa_a", "visa_c"]
    assert reg.vram_in_use() == 200
    assert reg.acquire("visa_a") is reg.acquire("visa_a")  # resident: the same session


def test_a_model_larger_than_the_budget_still_loads_alone(settings: ServeSettings) -> None:
    s = settings.model_copy(update={"vram_budget_mib": 50})
    reg = ModelRegistry(s)
    reg.acquire("visa_a")
    reg.acquire("visa_b")
    assert [m.product for m in reg.loaded()] == ["visa_b"]


def test_preload_loads_at_start_up(settings: ServeSettings) -> None:
    s = settings.model_copy(update={"preload": ["visa_a"]})
    with TestClient(create_app(s)) as c:
        assert [m["product"] for m in c.get("/health").json()["loaded"]] == ["visa_a"]


def test_an_edited_frozen_file_stops_the_start(settings: ServeSettings) -> None:
    p = settings.frozen_thresholds
    p.write_text(p.read_text().replace("0.4", "0.9", 1))
    with pytest.raises(ProtocolError, match="edited"):
        create_app(settings)


def test_create_session_on_cpu_reports_cpu(settings: ServeSettings) -> None:
    sess, provider = create_session(str(settings.products["steel"].onnx), False, [CPU], False)
    assert provider == CPU and sess.get_inputs()[0].name == "image"


# ---- the real config ------------------------------------------------------------------------


def test_default_config_parses_and_every_product_has_its_frozen_thresholds() -> None:
    from defectlens.eval.thresholds import read_frozen  # noqa: PLC0415

    cfg = load_settings(ROOT / "configs/serve/default.yaml")
    entries, _ = read_frozen(ROOT / cfg.frozen_thresholds)
    keys = {(e.domain, e.model, e.category): set() for e in entries}
    for e in entries:
        keys[(e.domain, e.model, e.category)].add(e.ratio)
    assert "steel" in cfg.products and any(p.startswith("visa_") for p in cfg.products)
    for name, spec in cfg.products.items():
        domain = "visa" if spec.kind == "anomaly" else "neu_det"
        assert keys[(domain, spec.frozen_model, spec.frozen_category)] == {5.0, 10.0, 50.0}, name
    assert cfg.fn_fp_ratio in {5.0, 10.0, 50.0}
    assert sum(p.vram_mib for p in cfg.products.values()) <= 2 * cfg.vram_budget_mib


def test_default_config_budget_holds_the_largest_two_products() -> None:
    cfg = load_settings(ROOT / "configs/serve/default.yaml")
    two = sorted((p.vram_mib for p in cfg.products.values()), reverse=True)[:2]
    assert sum(two) <= cfg.vram_budget_mib
