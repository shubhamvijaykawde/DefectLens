"""Image in -> defects, scores, boxes and heatmap (Phase 6a). numpy + OpenCV only, no torch.

Everything between the uploaded bytes and the JSON is here and is pure (the ONNX session is passed
in), so the same functions run in the API and in ``export/threshold_consistency.py``, which counts
how many selection-split images change side of a frozen threshold between PyTorch and this path.

* Steel (NEU-DET): grayscale decode -> ``Preprocessor`` (the training path: letterbox, 3 channels)
  -> detector -> boxes mapped back to the uploaded image -> keep ``score >= threshold``.
* VisA: colour decode -> ``resize_normalize`` (the training path) -> anomaly model ->
  ``flagged = score >= threshold``; regions = connected components of ``map >= threshold``.
  The regions are derived from the map and not evaluated as boxes (only pixel AUROC / AUPRO were).
"""

from __future__ import annotations

import base64
import struct
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from defectlens.data.neu_det import CLASSES
from defectlens.eval.detection import Detections
from defectlens.preprocess.anomaly_input import resize_normalize
from defectlens.preprocess.geometry import Affine, clip_boxes
from defectlens.preprocess.pipeline import PreprocessConfig, Preprocessor
from defectlens.serve.config import DFineSettings, ProductSpec, ServeSettings
from defectlens.serve.model_registry import LoadedModel, ModelRegistry
from defectlens.serve.schemas import (
    AnomalyResult,
    Defect,
    InspectResponse,
    OperatingPoint,
    Timing,
)

F32 = npt.NDArray[np.float32]
Image = npt.NDArray[np.uint8]

JPEG_SOF = frozenset(set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC})


class BadImage(ValueError):
    """The upload is not an image this service accepts (the API answers 400)."""


# --------------------------------------------------------------------------------------------
# Decoding


def sniff_image(data: bytes) -> tuple[str, int | None, int | None]:
    """Format and declared size of a JPEG / PNG / BMP upload, from its header bytes.

    The size is read before decoding so that a small file declaring a huge image (a decompression
    bomb) is refused without allocating it. BMP is uncompressed, so the byte limit bounds it.
    """
    if data[:3] == b"\xff\xd8\xff":
        pos = 2
        while pos + 4 <= len(data):
            if data[pos] != 0xFF:
                break
            marker = data[pos + 1]
            if marker == 0xFF:  # fill byte
                pos += 1
                continue
            if marker in JPEG_SOF:
                if pos + 9 > len(data):
                    break
                h, w = struct.unpack(">HH", data[pos + 5 : pos + 9])
                return "jpeg", int(w), int(h)
            if marker in (0x01, *range(0xD0, 0xDA)):  # markers without a length field
                pos += 2
                continue
            pos += 2 + struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        return "jpeg", None, None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if len(data) < 24:
            msg = "truncated PNG header"
            raise BadImage(msg)
        w, h = struct.unpack(">II", data[16:24])
        return "png", int(w), int(h)
    if data[:2] == b"BM":
        return "bmp", None, None
    msg = "unsupported image format: upload a JPEG, PNG or BMP file"
    raise BadImage(msg)


def decode_image(data: bytes, *, color: bool, max_pixels: int) -> Image:
    """Decode an upload to uint8: BGR ``(H, W, 3)`` if ``color`` else grayscale ``(H, W)``."""
    if not data:
        msg = "empty upload"
        raise BadImage(msg)
    _, w, h = sniff_image(data)
    if w is not None and h is not None and w * h > max_pixels:
        msg = f"image declares {w}x{h} pixels, above the limit of {max_pixels}"
        raise BadImage(msg)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flag)
    if img is None or img.size == 0:
        msg = "the file could not be decoded as an image (corrupt or truncated)"
        raise BadImage(msg)
    if img.shape[0] * img.shape[1] > max_pixels:
        msg = f"image has {img.shape[1]}x{img.shape[0]} pixels, above the limit of {max_pixels}"
        raise BadImage(msg)
    return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------------------------------
# Steel: preprocessing and detector output decoding


def preprocess_steel(
    gray: Image, steel_preprocess: dict[str, Any], size: int
) -> tuple[F32, Affine]:
    """Grayscale ``(H, W)`` -> ``(3, size, size)`` float32 in [0, 1] and the original->input map.

    The same ``Preprocessor`` the detectors were trained and scored with (``NeuDetDataset``).
    """
    pre = Preprocessor(PreprocessConfig.from_dict({**steel_preprocess, "size": size}))
    res = pre(gray)
    x = np.ascontiguousarray(res.image.transpose(2, 0, 1)).astype(np.float32) / 255.0
    return np.asarray(x, dtype=np.float32), res.affine


def decode_dfine(
    logits: F32, boxes: F32, size: int, cfg: DFineSettings
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """numpy twin of ``detection.dfine.decode`` for one image: ``logits (Q, K)``, ``boxes (Q, 4)``
    normalised cxcywh -> top-``top_k`` (query, class) pairs as xyxy pixels in the input frame,
    scores, labels 1..K. ``tests/test_serve_inference.py`` compares it with the torch version."""
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    n_q, n_c = probs.shape
    flat = probs.reshape(-1)
    k = min(cfg.top_k, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx], kind="stable")]
    scores = flat[idx]
    query, cls = idx // n_c, idx % n_c
    cxcywh = boxes[query].astype(np.float32)
    xyxy = np.concatenate([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], 1)
    xyxy = xyxy * np.array([size, size, size, size], dtype=np.float32)
    keep = scores >= cfg.score_floor
    return (
        xyxy[keep].astype(np.float64), scores[keep].astype(np.float64),
        (cls[keep] + 1).astype(np.int64),
    )


def to_original_frame(
    boxes: npt.NDArray[np.float64], scores: npt.ArrayLike, labels: npt.ArrayLike, affine: Affine,
    orig_hw: tuple[int, int],
) -> Detections:
    """Boxes in the network input frame -> the uploaded image's pixels, clipped to it."""
    h, w = orig_hw
    return Detections.of(clip_boxes(affine.invert(boxes), w, h), scores, labels)


def keep_at_or_above(dets: Detections, threshold: float) -> Detections:
    """The detections with ``score >= threshold`` (the rule of the frozen cost thresholds)."""
    k = dets.scores >= threshold
    return Detections(dets.boxes[k], dets.scores[k], dets.labels[k])


def class_name(label: int) -> str:
    return CLASSES[label - 1] if 1 <= label <= len(CLASSES) else f"class_{label}"


# --------------------------------------------------------------------------------------------
# VisA: preprocessing, regions and heatmap


def preprocess_visa(bgr: Image, image_size: tuple[int, int]) -> F32:
    """BGR ``(H0, W0, 3)`` -> ``(1, 3, H, W)`` float32, exactly ``VisaImages`` plus a batch axis."""
    x = resize_normalize(bgr, image_size)
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None].astype(np.float32)


def anomaly_regions(
    amap: F32, threshold: float, orig_hw: tuple[int, int], max_regions: int
) -> list[Defect]:
    """Connected regions of ``amap >= threshold`` as boxes in the uploaded image's pixels.

    The score of a region is the largest map value inside it. Not evaluated against the ground
    truth masks (the project measured pixel AUROC and AUPRO only).
    """
    mh, mw = amap.shape
    mask = (amap >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = orig_hw
    sx, sy = w / mw, h / mh
    out: list[Defect] = []
    for i in range(1, n):
        x, y, bw, bh, _ = (int(v) for v in stats[i])
        score = float(amap[labels == i].max())
        out.append(Defect(
            label="anomaly", score=score,
            box=(x * sx, y * sy, (x + bw) * sx, (y + bh) * sy),
        ))
    out.sort(key=lambda d: -d.score)
    return out[:max_regions]


def render_heatmap(
    bgr: Image, amap: F32, threshold: float, *, max_side: int, alpha: float
) -> bytes:
    """PNG overlay of the anomaly map on the image (longest side at most ``max_side``).

    Colour = JET of ``map / (2 * threshold)`` clipped to [0, 1]: the threshold sits mid-scale, so
    a region is yellow-to-red exactly where it would be flagged.
    """
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    out_w, out_h = max(1, round(w * s)), max(1, round(h * s))
    base = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    m = cv2.resize(amap, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    scale = 2.0 * max(threshold, 1e-12)
    level = (np.clip(m / scale, 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(level, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(heat, alpha, base, 1.0 - alpha, 0.0)
    ok, png = cv2.imencode(".png", blended)
    if not ok:
        msg = "PNG encoding of the heatmap failed"
        raise RuntimeError(msg)
    return bytes(png.tobytes())


# --------------------------------------------------------------------------------------------
# Running the models


def prepare_detector_input(
    spec: ProductSpec, settings: ServeSettings, gray: Image, size: int
) -> tuple[F32, Affine]:
    """The network input of one grayscale image (batch axis only for the D-FINE contract)."""
    x, affine = preprocess_steel(gray, settings.steel_preprocess, size)
    return (x[None] if spec.format == "dfine" else x), affine


def run_detector_onnx(
    loaded: LoadedModel, spec: ProductSpec, settings: ServeSettings, x: F32, affine: Affine,
    orig_hw: tuple[int, int],
) -> Detections:
    """All detections (no threshold) for a prepared input, in the uploaded image's pixels."""
    if spec.format == "dfine":
        logits, boxes = loaded.session.run(None, {loaded.input_name: x})[:2]
        b, s, lab = decode_dfine(
            np.asarray(logits, dtype=np.float32)[0], np.asarray(boxes, dtype=np.float32)[0],
            loaded.input_hw[0], settings.dfine,
        )
    else:  # torchvision: post-processing (score floor, NMS, top 100) is inside the graph
        boxes, scores, labels = loaded.session.run(None, {loaded.input_name: x})[:3]
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        s, lab = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    return to_original_frame(b, s, lab, affine, orig_hw)


def run_anomaly_onnx(loaded: LoadedModel, x: F32) -> tuple[float, F32]:
    """Image score and anomaly map ``(H, W)`` for a prepared ``(1, 3, H, W)`` input."""
    score, amap = loaded.session.run(None, {loaded.input_name: x})[:2]
    return float(np.asarray(score).reshape(-1)[0]), np.asarray(amap, dtype=np.float32)[0]


# --------------------------------------------------------------------------------------------
# One request


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _box4(b: npt.NDArray[np.float64]) -> tuple[float, float, float, float]:
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


@dataclass
class Inspector:
    """Ties settings, registry and the pure functions above together for one API process."""

    settings: ServeSettings
    registry: ModelRegistry

    def inspect(
        self, product: str, data: bytes, *, fn_fp_ratio: float | None = None, heatmap: bool = True
    ) -> InspectResponse:
        t_all = time.perf_counter()
        spec = self.registry.spec(product)  # UnknownProduct
        ratio = self.settings.fn_fp_ratio if fn_fp_ratio is None else fn_fp_ratio
        entry = self.registry.threshold(product, ratio)  # UnknownRatio
        t0 = time.perf_counter()
        loaded = self.registry.acquire(product)  # ModelUnavailable
        load_ms = _ms(t0)

        t0 = time.perf_counter()
        img = decode_image(data, color=spec.kind == "anomaly", max_pixels=self.settings.max_pixels)
        decode_ms = _ms(t0)
        orig_hw = (img.shape[0], img.shape[1])
        warnings: list[str] = []
        if spec.trained_image_size is not None and orig_hw != spec.trained_image_size:
            warnings.append(
                f"image is {orig_hw[1]}x{orig_hw[0]} px; the model was trained on "
                f"{spec.trained_image_size[1]}x{spec.trained_image_size[0]} px images, "
                "so scores outside that scale are untested"
            )

        defects: list[Defect] = []
        anomaly: AnomalyResult | None = None
        if spec.kind == "detector":
            t0 = time.perf_counter()
            x, affine = prepare_detector_input(spec, self.settings, img, loaded.input_hw[0])
            pre_ms = _ms(t0)
            t0 = time.perf_counter()
            dets = run_detector_onnx(loaded, spec, self.settings, x, affine, orig_hw)
            infer_ms = _ms(t0)
            t0 = time.perf_counter()
            kept = keep_at_or_above(dets, entry.threshold)
            defects = [
                Defect(
                    label=class_name(int(kept.labels[i])), score=float(kept.scores[i]),
                    box=_box4(kept.boxes[i]),
                )
                for i in np.argsort(-kept.scores, kind="stable")
            ]
            flagged = bool(defects)
        else:
            t0 = time.perf_counter()
            xa = preprocess_visa(img, loaded.input_hw)
            pre_ms = _ms(t0)
            t0 = time.perf_counter()
            score, amap = run_anomaly_onnx(loaded, xa)
            infer_ms = _ms(t0)
            t0 = time.perf_counter()
            flagged = score >= entry.threshold
            png_b64 = None
            if heatmap:
                png = render_heatmap(
                    img, amap, entry.threshold,
                    max_side=self.settings.heatmap.max_side, alpha=self.settings.heatmap.alpha,
                )
                png_b64 = base64.b64encode(png).decode("ascii")
            if flagged:
                defects = anomaly_regions(
                    amap, entry.threshold, orig_hw, self.settings.heatmap.max_regions
                )
            anomaly = AnomalyResult(
                score=score, threshold=entry.threshold, flagged=flagged,
                map_size=(int(amap.shape[0]), int(amap.shape[1])), heatmap_png_base64=png_b64,
            )
        post_ms = _ms(t0)
        return InspectResponse(
            product=product, model=spec.frozen_model, image_size=orig_hw, flagged=flagged,
            defects=defects, anomaly=anomaly,
            operating_point=OperatingPoint(
                fn_fp_ratio=entry.ratio, threshold=entry.threshold,
                frozen_file_sha256=self.registry.frozen_sha256,
            ),
            provider=loaded.provider, tf32=loaded.tf32, warnings=warnings,
            timing=Timing(
                decode_ms=decode_ms, preprocess_ms=pre_ms, inference_ms=infer_ms,
                postprocess_ms=post_ms, load_ms=load_ms, total_ms=_ms(t_all),
            ),
        )
