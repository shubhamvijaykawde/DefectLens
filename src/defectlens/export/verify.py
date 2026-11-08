"""Compare ONNX Runtime with PyTorch (Phase 5a): raw outputs, detection metrics, anomaly scores.

Reference = PyTorch on CUDA in strict float32 (``strict_fp32``: TF32 off in cuDNN and in matmul;
PyTorch's cuDNN default is TF32 on, which is itself a variable here). Everything else is compared
against it:

* D-FINE: raw logits / boxes, how many of the top-100 (query, class) pairs agree, and the
  detection metrics (AP) after the *same* decode step as the PyTorch path, so any metric
  difference comes from the network outputs and nothing else. Variants: PyTorch with the cuDNN
  default, PyTorch AMP (the Phase 2 path), ORT CUDA with TF32 off and on.
* Anomaly models: image scores (max abs / relative difference, rank agreement, AUROC) and the
  largest anomaly-map difference, on ``threshold_calibration`` (the selection split).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch
from scipy.stats import spearmanr

from defectlens.detection import dfine
from defectlens.detection.datamodule import NeuDetDataset
from defectlens.eval.anomaly import auroc
from defectlens.eval.detection import Detections, GroundTruth, evaluate_detections

F32 = npt.NDArray[np.float32]
RawRunner = Callable[[F32], tuple[F32, F32]]  # (1,3,S,S) -> (logits (1,Q,K), boxes (1,Q,4))


# --------------------------------------------------------------------------------------------
# Sessions and precision switches


def make_session(path: str, *, tf32: bool, device: str = "cuda") -> ort.InferenceSession:
    """ORT session on CUDA (``use_tf32`` set explicitly) or CPU; fails if CUDA did not load."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if device == "cpu":
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    providers: list[Any] = [
        ("CUDAExecutionProvider", {"use_tf32": "1" if tf32 else "0"}), "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(path, so, providers=providers)
    if sess.get_providers()[0] != "CUDAExecutionProvider":
        msg = f"CUDAExecutionProvider did not load for {path}: {sess.get_providers()}"
        raise RuntimeError(msg)
    return sess


@contextmanager
def strict_fp32() -> Iterator[None]:
    """TF32 off for cuDNN convolutions and matmuls inside the block (restored afterwards)."""
    conv, mm = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = conv, mm


# --------------------------------------------------------------------------------------------
# Pure comparison helpers (numpy only; tested on CPU)


def compare_arrays(ref: npt.ArrayLike, got: npt.ArrayLike) -> dict[str, float]:
    """Max / mean absolute difference and the largest relative one, |d| / max(|ref|, 1e-6)."""
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(got, dtype=np.float64)
    if a.shape != b.shape:
        msg = f"shape mismatch {a.shape} vs {b.shape}"
        raise ValueError(msg)
    d = np.abs(a - b)
    return {
        "max_abs": float(d.max()), "mean_abs": float(d.mean()),
        "max_rel": float((d / np.maximum(np.abs(a), 1e-6)).max()),
    }


def topk_overlap(ref_logits: F32, got_logits: F32, k: int = 100) -> float:
    """Mean over images of |top-k(ref) & top-k(got)| / k on the flattened (query, class) scores.

    1.0 means every image keeps exactly the same ``k`` candidate detections.
    """
    n = ref_logits.shape[0]
    a = np.argsort(-ref_logits.reshape(n, -1), axis=1)[:, :k]
    b = np.argsort(-got_logits.reshape(n, -1), axis=1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b, strict=True)]))


def raw_agreement(
    ref: RawOutputs, got: RawOutputs, *, atol: float, ks: Sequence[int] = (10, 100)
) -> dict[str, float]:
    """How well raw D-FINE outputs agree, in ways that survive a permutation of the 300 slots.

    ``logits`` / ``boxes`` are slot-wise max abs differences over everything; a single image whose
    top-k proposal selection flips makes them large although the detections are the same set, so
    ``images_within_atol`` counts the images where both stay under ``atol``. ``sorted_prob_max_abs``
    is the largest difference between the descending-sorted sigmoid scores of the top 100
    (query, class) pairs, which ignores which slot holds which detection. ``top<k>_overlap`` is
    slot-aware (a detection that moved to another query counts as a miss): the share of the
    reference's top-k (query, class) pairs that are also in the other's top-k.
    """
    n = ref.logits.shape[0]
    per_image = np.maximum(
        np.abs(ref.logits - got.logits).reshape(n, -1).max(axis=1),
        np.abs(ref.boxes - got.boxes).reshape(n, -1).max(axis=1),
    )
    sig = 1.0 / (1.0 + np.exp(-ref.logits.reshape(n, -1).astype(np.float64)))
    sig_got = 1.0 / (1.0 + np.exp(-got.logits.reshape(n, -1).astype(np.float64)))
    top = min(100, sig.shape[1])
    s_ref = -np.sort(-sig, axis=1)[:, :top]
    s_got = -np.sort(-sig_got, axis=1)[:, :top]
    out = {
        "logits_max_abs": float(np.abs(ref.logits - got.logits).max()),
        "boxes_max_abs": float(np.abs(ref.boxes - got.boxes).max()),
        "images_within_atol": float((per_image <= atol).sum()),
        "n_images": float(n),
        "sorted_prob_max_abs": float(np.abs(s_ref - s_got).max()),
    }
    for k in ks:
        out[f"top{k}_overlap"] = topk_overlap(ref.logits, got.logits, k)
    return out


def rank_agreement(ref: npt.ArrayLike, got: npt.ArrayLike) -> float:
    """Spearman correlation of two score vectors (1.0 = identical ranking)."""
    r = np.asarray(ref, dtype=np.float64)
    g = np.asarray(got, dtype=np.float64)
    if np.array_equal(r, g):
        return 1.0
    return float(spearmanr(r, g).statistic)


# --------------------------------------------------------------------------------------------
# D-FINE: raw outputs, detections, metrics


@dataclass
class RawOutputs:
    logits: F32  # (N, Q, K)
    boxes: F32  # (N, Q, 4)


def collect_raw(
    ds: NeuDetDataset, runners: Mapping[str, RawRunner]
) -> tuple[dict[str, RawOutputs], list[dict[str, Any]]]:
    """Run every variant on every image of ``ds`` (one data pass, batch 1)."""
    logits: dict[str, list[F32]] = {k: [] for k in runners}
    boxes: dict[str, list[F32]] = {k: [] for k in runners}
    metas: list[dict[str, Any]] = []
    for i in range(len(ds)):
        img, _, meta = ds[i]
        x = img.unsqueeze(0).numpy().astype(np.float32)
        for name, run in runners.items():
            lg, bx = run(x)
            logits[name].append(lg)
            boxes[name].append(bx)
        metas.append(meta)
    raw = {k: RawOutputs(np.concatenate(logits[k]), np.concatenate(boxes[k])) for k in runners}
    return raw, metas


def torch_runner(
    model: torch.nn.Module, device: torch.device, *, amp: bool = False
) -> RawRunner:
    def run(x: F32) -> tuple[F32, F32]:
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(pixel_values=torch.from_numpy(x).to(device))
        return (
            out.logits.float().cpu().numpy().astype(np.float32),
            out.pred_boxes.float().cpu().numpy().astype(np.float32),
        )

    return run


def ort_runner(sess: ort.InferenceSession) -> RawRunner:
    name = sess.get_inputs()[0].name

    def run(x: F32) -> tuple[F32, F32]:
        logits, boxes = sess.run(None, {name: x})
        return np.asarray(logits, dtype=np.float32), np.asarray(boxes, dtype=np.float32)

    return run


def detections_from_raw(
    raw: RawOutputs, metas: Sequence[Mapping[str, Any]], size: int, *,
    score_thresh: float, detections_per_img: int,
) -> list[Detections]:
    """The PyTorch path's decode + mapping back to the original frame, applied to ``raw``."""
    decoded = dfine.decode(
        torch.from_numpy(raw.logits), torch.from_numpy(raw.boxes), (size, size),
        score_thresh=score_thresh, detections_per_img=detections_per_img,
    )
    return [dfine.to_detections(*d, dict(m)) for d, m in zip(decoded, metas, strict=True)]


def detection_metrics(
    dets: Sequence[Detections], gts: Sequence[GroundTruth], class_names: Mapping[int, str],
    recall_score_thr: float,
) -> dict[str, float]:
    return evaluate_detections(dets, gts, class_names, recall_score_thr=recall_score_thr)


def bootstrap_differences(
    dets: Mapping[str, Sequence[Detections]], gts: Sequence[GroundTruth],
    class_names: Mapping[int, str], pairs: Mapping[str, tuple[str, str]], keys: Sequence[str],
    *, n_boot: int, seed: int, recall_score_thr: float = 0.5,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """For each named pair ``(a, b)`` and metric: ``metric(b) - metric(a)`` on the full set and a
    paired image-bootstrap 95% interval, as ``(difference, low, high)``.

    The same resampled images are used for every variant, so an interval measures the effect of
    the variant, not the sampling noise of the images. Each variant is evaluated once per
    resample (an evaluation costs about 0.3 s on 270 images).
    """
    rng = np.random.default_rng(seed)
    n = len(gts)
    names = sorted({v for pair in pairs.values() for v in pair})

    def metrics(idx: npt.NDArray[np.intp]) -> dict[str, dict[str, float]]:
        return {
            v: evaluate_detections(
                [dets[v][i] for i in idx], [gts[i] for i in idx], class_names,
                recall_score_thr=recall_score_thr,
            )
            for v in names
        }

    point = metrics(np.arange(n))
    draws: list[dict[str, dict[str, float]]] = []
    for _ in range(n_boot):
        draws.append(metrics(rng.integers(0, n, size=n)))
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for label, (a, b) in pairs.items():
        out[label] = {}
        for k in keys:
            diffs = [d[b][k] - d[a][k] for d in draws]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            out[label][k] = (float(point[b][k] - point[a][k]), float(lo), float(hi))
    return out


# --------------------------------------------------------------------------------------------
# Anomaly models


def anomaly_agreement(
    ref_scores: npt.ArrayLike, got_scores: npt.ArrayLike, labels: npt.ArrayLike,
    max_map_diff: float,
) -> dict[str, float]:
    """Score agreement between two runs of one anomaly model on a labelled split."""
    out = compare_arrays(ref_scores, got_scores)
    out["spearman"] = rank_agreement(ref_scores, got_scores)
    out["auroc_ref"] = auroc(ref_scores, labels)
    out["auroc_got"] = auroc(got_scores, labels)
    out["max_map_abs"] = max_map_diff
    return out
