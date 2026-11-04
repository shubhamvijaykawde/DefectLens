import numpy as np
import pytest

from defectlens.preprocess.clahe import apply_clahe, to_three_channel
from defectlens.preprocess.geometry import Affine
from defectlens.preprocess.pipeline import PreprocessConfig, Preprocessor
from defectlens.preprocess.resize import resize_to_square
from defectlens.preprocess.roi import crop_roi


def _scene(h: int = 100, w: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Gray image with a bright rectangle; returns (image, box xyxy of the rectangle)."""
    img = np.full((h, w), 50, dtype=np.uint8)
    img[20:60, 40:120] = 250
    return img, np.array([[40, 20, 120, 60]], dtype=float)


def _bright_bbox(img: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(img[..., 0] > 200 if img.ndim == 3 else img > 200)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def test_affine_roundtrip_and_composition() -> None:
    a, b = Affine(0.5, 3, 4), Affine(2.0, -1, 7)
    boxes = np.array([[10.0, 20.0, 30.0, 50.0]])
    np.testing.assert_allclose(a.then(b).apply(boxes), b.apply(a.apply(boxes)))
    np.testing.assert_allclose(a.invert(a.apply(boxes)), boxes)


def test_letterbox_box_stays_on_object() -> None:
    img, box = _scene()
    out, aff = resize_to_square(img, 64)
    assert out.shape == (64, 64)
    mapped = aff.apply(box)[0]
    x0, y0, x1, y1 = _bright_bbox(out)
    np.testing.assert_allclose(mapped, [x0, y0, x1, y1], atol=1.0)
    assert (out[0, 0], out[-1, -1]) == (114, 114)  # padded rows
    assert out.dtype == np.uint8


def test_upscale_letterbox_alignment() -> None:
    img = np.full((50, 100), 50, dtype=np.uint8)
    img[10:30, 20:60] = 250
    box = np.array([[20, 10, 60, 30]], dtype=float)
    out, aff = resize_to_square(img, 400)
    np.testing.assert_allclose(aff.apply(box)[0], _bright_bbox(out), atol=2.0)


def test_stretch_rejected() -> None:
    with pytest.raises(ValueError, match="letterbox"):
        resize_to_square(np.zeros((4, 8), np.uint8), 16, letterbox=False)


def test_crop_remaps_clips_and_drops() -> None:
    img = np.zeros((100, 200), np.uint8)
    boxes = np.array(
        [
            [50, 10, 90, 40],  # fully inside
            [0, 0, 60, 30],  # 1/6 of its width inside the ROI -> dropped
            [90, 30, 130, 45],  # half outside on the right -> kept, clipped
            [150, 70, 190, 90],  # fully outside -> dropped
        ],
        dtype=float,
    )
    crop, out, keep, shift = crop_roi(img, boxes, (50, 0, 110, 50), min_visible=0.5)
    assert crop.shape == (50, 60)
    assert keep.tolist() == [True, False, True, False]
    np.testing.assert_allclose(out, [[0, 10, 40, 40], [40, 30, 60, 45]])
    assert shift == Affine(1.0, -50.0, 0.0)


def test_crop_invalid_roi() -> None:
    with pytest.raises(ValueError, match="invalid"):
        crop_roi(np.zeros((10, 10), np.uint8), np.zeros((0, 4)), (0, 0, 11, 5))


def test_grayscale_to_three_channel() -> None:
    g = np.arange(12, dtype=np.uint8).reshape(3, 4)
    for src in (g, g[:, :, None]):
        out = to_three_channel(src)
        assert out.shape == (3, 4, 3)
        for c in range(3):
            np.testing.assert_array_equal(out[:, :, c], g)
    bgr = np.zeros((3, 4, 3), np.uint8)
    assert to_three_channel(bgr) is bgr


def test_clahe_deterministic_and_changes_contrast() -> None:
    rng = np.random.default_rng(0)
    img = (rng.normal(100, 5, (64, 64)).clip(0, 255)).astype(np.uint8)
    a, b = apply_clahe(img), apply_clahe(img.copy())
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.uint8 and a.shape == img.shape
    assert a.std() > img.std()  # low-contrast input gets stretched
    color = np.stack([img] * 3, axis=2)
    assert apply_clahe(color).shape == color.shape
    with pytest.raises(ValueError, match="uint8"):
        apply_clahe(img.astype(np.float32))


def test_pipeline_boxes_aligned_and_invertible() -> None:
    img, box = _scene(100, 200)
    for roi, clahe in [(None, False), ((20, 10, 160, 90), False), (None, True)]:
        cfg = PreprocessConfig(size=128, roi=roi, clahe=clahe)
        res = Preprocessor(cfg)(img, box)
        assert res.image.shape == (128, 128, 3)
        if not clahe:  # CLAHE rescales intensities, so the >200 threshold is only valid without
            np.testing.assert_allclose(res.boxes[0], _bright_bbox(res.image), atol=1.5)
        np.testing.assert_allclose(res.to_original(res.boxes), box, atol=1e-9)
        assert res.original_size == (100, 200)


def test_pipeline_without_boxes_and_dropped_boxes() -> None:
    img, box = _scene()
    p = Preprocessor(PreprocessConfig(size=64, roi=(150, 0, 200, 100)))
    res = p(img, box)  # the rectangle lies outside the ROI
    assert res.boxes.shape == (0, 4) and not res.keep.any()
    assert Preprocessor(PreprocessConfig(size=64))(img).boxes.shape == (0, 4)


def test_config_from_dict() -> None:
    cfg = PreprocessConfig.from_dict(
        {
            "size": 480,
            "clahe": {"enabled": True, "clip_limit": 3.0, "tile_grid": [4, 4]},
            "roi": [0, 0, 10, 10],
        }
    )
    assert cfg.size == 480 and cfg.clahe and cfg.clahe_tile_grid == (4, 4)
    assert cfg.roi == (0, 0, 10, 10)
