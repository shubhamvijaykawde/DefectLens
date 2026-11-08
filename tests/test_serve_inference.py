"""The pure parts of the API (Phase 6a): decoding, preprocessing parity with training, decoding of
the detector outputs, regions and heatmap. CPU only, tiny inputs."""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from defectlens.anomaly.datamodule import VisaImages, load_split_paths
from defectlens.detection import dfine
from defectlens.detection.datamodule import NeuDetDataset
from defectlens.eval.detection import Detections
from defectlens.preprocess.geometry import Affine
from defectlens.preprocess.pipeline import PreprocessConfig, Preprocessor
from defectlens.serve import inference as inf
from defectlens.serve.config import DFineSettings, load_settings

ROOT = Path(__file__).resolve().parents[1]


def png_bytes(h: int = 8, w: int = 12, color: bool = True) -> bytes:
    img = np.full((h, w, 3) if color else (h, w), 90, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf.tobytes())


def jpeg_bytes(h: int, w: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w), 100, dtype=np.uint8))
    assert ok
    return bytes(buf.tobytes())


# ---- decoding -------------------------------------------------------------------------------


def test_sniff_reads_declared_size_of_png_and_jpeg() -> None:
    assert inf.sniff_image(png_bytes(8, 12)) == ("png", 12, 8)
    assert inf.sniff_image(jpeg_bytes(30, 50)) == ("jpeg", 50, 30)
    assert inf.sniff_image(b"BM" + b"\x00" * 20)[0] == "bmp"


@pytest.mark.parametrize("data", [b"", b"hello world", b"GIF89a......", b"\x89PNG\r\n\x1a\n"])
def test_sniff_and_decode_refuse_non_images(data: bytes) -> None:
    with pytest.raises(inf.BadImage):
        inf.decode_image(data, color=True, max_pixels=10**6)


def test_decode_color_and_gray_shapes() -> None:
    assert inf.decode_image(png_bytes(8, 12), color=True, max_pixels=10**6).shape == (8, 12, 3)
    assert inf.decode_image(png_bytes(8, 12), color=False, max_pixels=10**6).shape == (8, 12)


def test_declared_size_above_limit_is_refused_before_decoding() -> None:
    bomb = bytearray(png_bytes())
    bomb[16:24] = struct.pack(">II", 60000, 60000)  # a tiny file that claims 3.6e9 pixels
    with pytest.raises(inf.BadImage, match="above the limit"):
        inf.decode_image(bytes(bomb), color=True, max_pixels=40_000_000)
    with pytest.raises(inf.BadImage, match="above the limit"):
        inf.decode_image(jpeg_bytes(100, 100), color=True, max_pixels=5_000)


def test_corrupt_body_with_a_valid_header_is_refused() -> None:
    with pytest.raises(inf.BadImage, match="could not be decoded"):
        inf.decode_image(png_bytes()[:40], color=True, max_pixels=10**6)


# ---- parity with the training / evaluation path --------------------------------------------


def test_steel_preprocessing_equals_the_training_dataset(tiny_data: tuple[Path, Path]) -> None:
    raw, processed = tiny_data
    cfg = yaml.safe_load((ROOT / "configs/serve/default.yaml").read_text())["steel_preprocess"]
    pre = Preprocessor(PreprocessConfig.from_dict({**cfg, "size": 64}))
    ds = NeuDetDataset(processed / "val.json", raw / "IMAGES", pre)
    for i, item in enumerate(ds.items):
        gray = inf.decode_image((raw / "IMAGES" / item.file_name).read_bytes(),
                                color=False, max_pixels=10**6)
        x, affine = inf.preprocess_steel(gray, cfg, 64)
        ref, _, meta = ds[i]
        assert x.dtype == np.float32 and x.shape == (3, 64, 64)
        np.testing.assert_array_equal(x, ref.numpy())
        assert affine == meta["affine"]


def test_visa_preprocessing_equals_visaimages(tiny_visa: dict[str, Path]) -> None:
    paths = load_split_paths(tiny_visa["splits"], "cat", "threshold_calibration")
    ds = VisaImages(tiny_visa["root"], tiny_visa["csv"], "cat", paths, (32, 40))
    for i, rel in enumerate(paths):
        bgr = inf.decode_image((tiny_visa["root"] / rel).read_bytes(), color=True,
                               max_pixels=10**6)
        x = inf.preprocess_visa(bgr, (32, 40))
        assert x.shape == (1, 3, 32, 40) and x.dtype == np.float32
        np.testing.assert_array_equal(x[0], ds[i]["image"].numpy())


def test_serve_steel_preprocess_matches_the_preprocess_config() -> None:
    served = load_settings(ROOT / "configs/serve/default.yaml").steel_preprocess
    base = yaml.safe_load((ROOT / "configs/preprocess/default.yaml").read_text())
    assert PreprocessConfig.from_dict({**served, "size": 1}) == PreprocessConfig.from_dict(
        {**base, "size": 1}
    )


def test_serve_package_does_not_import_torch() -> None:
    code = (
        "import sys, defectlens.serve.app, defectlens.serve.inference, defectlens.serve.config;"
        "bad = sorted(m for m in ('torch', 'torchvision', 'transformers', 'anomalib', 'mlflow')"
        " if m in sys.modules); assert not bad, bad"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr


# ---- detector outputs -----------------------------------------------------------------------


def test_numpy_dfine_decode_matches_the_torch_decode() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(-2, 2, (20, 6)).astype(np.float32)
    boxes = rng.uniform(0.1, 0.9, (20, 4)).astype(np.float32)
    cfg = DFineSettings(top_k=15, score_floor=0.05)
    b, s, lab = inf.decode_dfine(logits, boxes, 64, cfg)
    (tb, ts, tl), = dfine.decode(
        torch.from_numpy(logits)[None], torch.from_numpy(boxes)[None], (64, 64),
        score_thresh=0.05, detections_per_img=15,
    )
    np.testing.assert_allclose(s, ts.numpy(), atol=1e-6)
    np.testing.assert_array_equal(lab, tl.numpy())
    np.testing.assert_allclose(b, tb.numpy(), atol=1e-4)
    assert (np.diff(s) <= 0).all()


def test_boxes_map_back_to_the_uploaded_image_through_the_letterbox() -> None:
    pre = Preprocessor(PreprocessConfig(size=64))
    res = pre(np.zeros((40, 80), dtype=np.uint8))  # padded top and bottom
    orig = np.array([[10.0, 5.0, 50.0, 30.0]])
    in_frame = res.affine.apply(orig)
    d = inf.to_original_frame(in_frame, [0.9], [3], res.affine, (40, 80))
    np.testing.assert_allclose(d.boxes, orig, atol=1e-9)
    clipped = inf.to_original_frame(np.array([[-5.0, -5.0, 900.0, 900.0]]), [1.0], [1],
                                    Affine(), (40, 80))
    np.testing.assert_array_equal(clipped.boxes, [[0, 0, 80, 40]])


def test_threshold_keeps_scores_equal_to_it() -> None:
    d = Detections.of(np.zeros((3, 4)), [0.9, 0.5, 0.2], [1, 2, 3])
    kept = inf.keep_at_or_above(d, 0.5)
    assert kept.scores.tolist() == [0.9, 0.5] and kept.labels.tolist() == [1, 2]


def test_class_names_follow_the_label_ids() -> None:
    assert inf.class_name(1) == "crazing" and inf.class_name(6) == "scratches"
    assert inf.class_name(99) == "class_99"


# ---- anomaly regions and heatmap ------------------------------------------------------------


def test_regions_are_boxes_of_the_map_above_threshold_in_image_pixels() -> None:
    amap = np.zeros((20, 30), dtype=np.float32)
    amap[2:6, 3:9] = 0.8
    amap[12:15, 20:28] = 0.6
    regions = inf.anomaly_regions(amap, 0.5, (40, 90), max_regions=10)
    assert [round(r.score, 2) for r in regions] == [0.8, 0.6]
    assert regions[0].box == pytest.approx((3 * 3.0, 2 * 2.0, 9 * 3.0, 6 * 2.0))
    assert len(inf.anomaly_regions(amap, 0.5, (40, 90), max_regions=1)) == 1
    assert inf.anomaly_regions(amap, 0.9, (40, 90), max_regions=10) == []


def test_heatmap_is_a_png_no_larger_than_max_side() -> None:
    bgr = np.full((200, 400, 3), 80, dtype=np.uint8)
    amap = np.random.default_rng(0).uniform(0, 1, (16, 32)).astype(np.float32)
    png = inf.render_heatmap(bgr, amap, 0.5, max_side=100, alpha=0.5)
    out = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (50, 100, 3)
    small = inf.render_heatmap(bgr[:20, :30], amap, 0.5, max_side=100, alpha=0.5)
    assert cv2.imdecode(np.frombuffer(small, np.uint8), cv2.IMREAD_COLOR).shape == (20, 30, 3)


def test_heatmap_threshold_is_mid_scale() -> None:
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    hot, cold = np.full((4, 4), 1.0, np.float32), np.zeros((4, 4), np.float32)
    ima = [cv2.imdecode(np.frombuffer(inf.render_heatmap(bgr, m, 0.5, max_side=10, alpha=1.0),
                                      np.uint8), cv2.IMREAD_COLOR) for m in (hot, cold)]
    assert ima[0][0, 0, 2] >= 120 and ima[0][0, 0, 0] < 10   # 2 * threshold: dark red (BGR)
    assert ima[1][0, 0, 0] >= 120 and ima[1][0, 0, 2] < 10   # 0: dark blue
