"""Tiny ONNX models and settings for the API tests (Phase 6a).

The models are hand-built graphs with fixed outputs (detectors) or a trivial function of the
input (anomaly), so a test knows exactly which boxes / scores the API must return. They follow
the ONNX contracts of ``export/onnx_export.py``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnx
from onnx import TensorProto, helper, numpy_helper

from defectlens.eval.thresholds import FrozenEntry, write_frozen
from defectlens.serve.config import ProductSpec, ServeSettings

CPU = "CPUExecutionProvider"
SIZE = 64          # detector input
ANOM_HW = (32, 48)  # anomaly input (height, width)


def _save(graph: onnx.GraphProto, path: Path) -> Path:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def _const(name: str, arr: npt.NDArray[np.generic]) -> onnx.NodeProto:
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(arr, name))


def torchvision_model(path: Path) -> Path:
    """Three boxes in the 64 px input frame -> in a 200 px square image: (20,40,100,80),
    (0,0,200,200), (50,50,150,150), scores .9 / .5 / .2, labels inclusion / scratches / crazing."""
    boxes = np.array([[6.4, 12.8, 32, 25.6], [0, 0, 64, 64], [16, 16, 48, 48]], np.float32)
    graph = helper.make_graph(
        [_const("boxes", boxes), _const("scores", np.array([0.9, 0.5, 0.2], np.float32)),
         _const("labels", np.array([2, 6, 1], np.int64))],
        "tiny_torchvision",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [3, SIZE, SIZE])],
        [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [3, 4]),
         helper.make_tensor_value_info("scores", TensorProto.FLOAT, [3]),
         helper.make_tensor_value_info("labels", TensorProto.INT64, [3])],
    )
    return _save(graph, path)


def dfine_model(path: Path) -> Path:
    """Three queries, six classes: query 0 is class 2 at p=.88, query 1 is class 5 at p=.5;
    everything else p=.018. Boxes (200 px image): (30,30,70,70) and (50,50,150,150)."""
    logits = np.full((1, 3, 6), -4.0, np.float32)
    logits[0, 0, 1], logits[0, 1, 4] = 2.0, 0.0
    boxes = np.array([[[0.25, 0.25, 0.2, 0.2], [0.5, 0.5, 0.5, 0.5], [0.9, 0.9, 0.1, 0.1]]],
                     np.float32)
    graph = helper.make_graph(
        [_const("logits", logits), _const("boxes", boxes)], "tiny_dfine",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3, 6]),
         helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 3, 4])],
    )
    return _save(graph, path)


def anomaly_model(path: Path) -> Path:
    """map = mean over channels of the normalised input, score = max of the map. A white pixel
    scores 2.44, a black one -1.99."""
    h, w = ANOM_HW
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["image"], ["anomaly_map"], axes=[1], keepdims=0),
         helper.make_node("ReduceMax", ["anomaly_map"], ["score"], axes=[1, 2], keepdims=0)],
        "tiny_anomaly",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, h, w])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [1]),
         helper.make_tensor_value_info("anomaly_map", TensorProto.FLOAT, [1, h, w])],
    )
    return _save(graph, path)


def frozen_file(path: Path) -> Path:
    def entry(domain: str, model: str, cat: str, ratio: float, t: float) -> FrozenEntry:
        split = "val" if domain == "neu_det" else "threshold_calibration"
        return FrozenEntry(domain, model, cat, ratio, t, split, 10, 5, 2, 1, None, 7.0)

    entries = []
    for model in ("tiny_tv", "tiny_df"):
        entries += [entry("neu_det", model, "all", r, t)
                    for r, t in ((5.0, 0.1), (10.0, 0.4), (50.0, 0.5))]
    entries += [entry("visa", "tiny_ad", "cat", r, t) for r, t in ((5.0, 3.0), (10.0, 0.5))]
    write_frozen(path, entries, {"iou": 0.5, "ratios": [5.0, 10.0, 50.0]})
    return path


def make_settings(tmp: Path, **overrides: object) -> ServeSettings:
    models = tmp / "models"
    steel_pre = {"letterbox": True, "pad_value": 114, "clahe": {"enabled": False},
                 "roi": None, "roi_min_visible": 0.5, "three_channel": True}
    anomaly = anomaly_model(models / "anomaly.onnx")

    def det(fmt: str, onnx_path: Path, model: str) -> ProductSpec:
        return ProductSpec(kind="detector", format=fmt, onnx=onnx_path, frozen_model=model,  # type: ignore[arg-type]
                           frozen_category="all", vram_mib=100, trained_image_size=(200, 200))

    def anom(onnx_path: Path) -> ProductSpec:
        return ProductSpec(kind="anomaly", format="anomaly", onnx=onnx_path,
                           frozen_model="tiny_ad", frozen_category="cat", vram_mib=100)

    products = {
        "steel": det("torchvision", torchvision_model(models / "tv.onnx"), "tiny_tv"),
        "steel_dfine": det("dfine", dfine_model(models / "df.onnx"), "tiny_df"),
        "visa_a": anom(anomaly), "visa_b": anom(anomaly), "visa_c": anom(anomaly),
        "visa_missing": anom(models / "not_there.onnx"),
    }
    base: dict[str, object] = {
        "frozen_thresholds": frozen_file(tmp / "frozen.json"), "fn_fp_ratio": 10.0,
        "providers": [CPU], "vram_budget_mib": 250, "steel_preprocess": steel_pre,
        "products": products,
    }
    return ServeSettings.model_validate({**base, **overrides})


def square_png(h: int = 64, w: int = 96) -> bytes:
    """Black image with a white block at rows 16-48, columns 24-72 (an exact 2x2 multiple)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[16:48, 24:72] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf.tobytes())


def steel_jpeg(size: int = 200) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((size, size), 120, np.uint8))
    assert ok
    return bytes(buf.tobytes())
