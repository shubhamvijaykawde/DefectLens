"""Run the PyTorch-vs-ONNX Runtime comparisons of Phase 5a and return plain-dict results.

Splits follow the Phase 4a protocol (``check_split``): D-FINE and Faster R-CNN are compared on
NEU-DET ``val`` (the selection split, where every decision is made) and, for D-FINE, also on
``test`` (the evaluation split, reported only: the TF32 default is chosen on ``val``); anomaly
models on VisA ``threshold_calibration``. Nothing here changes a threshold or a checkpoint.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from defectlens.anomaly.datamodule import VisaImages, load_split_paths
from defectlens.detection import faster_rcnn
from defectlens.detection.datamodule import NeuDetDataset, make_loader
from defectlens.eval.detection import Detections
from defectlens.eval.scoring import load_anomaly_model
from defectlens.eval.thresholds import check_split
from defectlens.export.onnx_export import load_detector
from defectlens.export.verify import (
    F32,
    RawRunner,
    anomaly_agreement,
    bootstrap_differences,
    collect_raw,
    detection_metrics,
    detections_from_raw,
    make_session,
    ort_runner,
    raw_agreement,
    strict_fp32,
    torch_runner,
)
from defectlens.preprocess.geometry import clip_boxes
from defectlens.preprocess.pipeline import PreprocessConfig, Preprocessor

SPLIT_PURPOSE = {"val": "selection", "test": "evaluation"}
REFERENCE = "torch_fp32_strict"
KEYS = ("map50", "map50_95", "recall", "precision", "fp_per_image", "recall_max")


def neu_dataset(mcfg: DictConfig, split: str) -> NeuDetDataset:
    """The NEU-DET split as the detector was trained and scored (like ``scoring.detect_neu``)."""
    pre_cfg: dict[str, Any] = {
        **OmegaConf.to_container(mcfg.preprocess),  # type: ignore[dict-item]
        "size": int(mcfg.model.input_size),
    }
    return NeuDetDataset(
        Path(mcfg.data.processed_dir) / f"{split}.json",
        Path(mcfg.data.raw_dir) / "IMAGES",
        Preprocessor(PreprocessConfig.from_dict(pre_cfg)),
    )


def verify_dfine(cfg: DictConfig, onnx_path: Path) -> dict[str, Any]:
    """Raw outputs and detection metrics of D-FINE: PyTorch variants against ORT TF32 off / on."""
    ex, v = cfg.export, cfg.verify
    model, mcfg, class_names = load_detector(Path(ex.detectors.dfine_s))
    device = torch.device("cuda")
    model.to(device)
    size = int(mcfg.model.input_size)
    score_thresh, per_img = float(mcfg.model.score_thresh), int(mcfg.model.detections_per_img)
    recall_thr = float(mcfg.model.recall_score_thr)

    def strict(run: RawRunner) -> RawRunner:
        def wrapped(x: F32) -> tuple[F32, F32]:
            with strict_fp32():
                return run(x)

        return wrapped

    runners: dict[str, RawRunner] = {
        REFERENCE: strict(torch_runner(model, device)),
        "torch_fp32_default": torch_runner(model, device),  # cuDNN TF32 convolutions as shipped
        "torch_amp": torch_runner(model, device, amp=True),  # the Phase 2 evaluation path
    }
    for tf32 in v.tf32:
        runners[f"ort_tf32_{'on' if tf32 else 'off'}"] = ort_runner(
            make_session(str(onnx_path), tf32=bool(tf32))
        )

    save_dir = Path(str(v.raw_dir)) if v.get("raw_dir") else None
    result: dict[str, Any] = {"splits": {}, "tolerance": float(v.raw_atol)}
    for split in v.dfine_splits:
        purpose = SPLIT_PURPOSE[split]
        check_split(purpose, "neu_det", split)
        ds = neu_dataset(mcfg, split)
        t0 = time.perf_counter()
        raw, metas = collect_raw(ds, runners)
        gts = ds.ground_truth(len(ds))
        dets = {
            name: detections_from_raw(
                r, metas, size, score_thresh=score_thresh, detections_per_img=per_img
            )
            for name, r in raw.items()
        }
        ref = raw[REFERENCE]
        variants: dict[str, Any] = {}
        metric_cache = {name: detection_metrics(dets[name], gts, class_names, recall_thr)
                        for name in raw}
        for name, r in raw.items():
            variants[name] = {
                "metrics": {k: metric_cache[name][k] for k in KEYS},
                "raw": raw_agreement(ref, r, atol=float(v.raw_atol), ks=(10, int(v.top_k))),
            }
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            for name, r in raw.items():
                np.savez_compressed(save_dir / f"dfine_{split}_{name}.npz",
                                    logits=r.logits, boxes=r.boxes)
        pairs = {  # the question of Phase 5a: does ORT's TF32 setting change the detection metrics?
            "ort_tf32_on_minus_ort_tf32_off": ("ort_tf32_off", "ort_tf32_on"),
        }
        boot = (
            bootstrap_differences(
                dets, gts, class_names, pairs, ["map50", "map50_95"],
                n_boot=int(v.bootstrap), seed=int(v.bootstrap_seed), recall_score_thr=recall_thr,
            )
            if all(x in dets for pair in pairs.values() for x in pair) else {}
        )
        result["splits"][split] = {
            "purpose": purpose, "n_images": len(ds), "variants": variants,
            "bootstrap_95": boot, "seconds": time.perf_counter() - t0,
        }
    return result


def verify_anomaly(
    cfg: DictConfig, model_name: str, category: str, onnx_path: Path
) -> dict[str, Any]:
    """Image scores and anomaly maps of one anomaly model: PyTorch (strict fp32) vs ORT."""
    v = cfg.verify
    split = str(v.anomaly_split)
    check_split("selection", "visa", split)
    obj, mcfg = load_anomaly_model(model_name, category, "cuda")
    data = mcfg.data
    size = (int(mcfg.model.image_size[0]), int(mcfg.model.image_size[1]))
    paths = load_split_paths(Path(data.splits_dir), category, split)
    ds = VisaImages(Path(data.raw_dir), Path(data.csv), category, paths, size)
    sessions = {bool(t): make_session(str(onnx_path), tf32=bool(t)) for t in v.tf32}

    ref_scores: list[float] = []
    got: dict[bool, list[float]] = {t: [] for t in sessions}
    max_map = dict.fromkeys(sessions, 0.0)
    for i in range(len(ds)):
        x = ds[i]["image"].unsqueeze(0)
        with strict_fp32():
            s, m = obj.predict(x.to("cuda"))
        ref_scores.append(float(s.float().cpu()[0]))
        ref_map = m.float().cpu().numpy()
        for t, sess in sessions.items():
            out_s, out_m = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
            got[t].append(float(np.asarray(out_s).reshape(-1)[0]))
            max_map[t] = max(max_map[t], float(np.abs(np.asarray(out_m) - ref_map).max()))
    labels = np.array(ds.labels)
    out: dict[str, Any] = {
        "model": model_name, "category": category, "split": split, "n_images": len(ds),
        "variants": {},
    }
    for t in sessions:
        out["variants"][f"ort_tf32_{'on' if t else 'off'}"] = anomaly_agreement(
            ref_scores, got[t], labels, max_map[t]
        )
    return out


def verify_faster_rcnn(cfg: DictConfig, onnx_path: Path) -> dict[str, Any]:
    """Faster R-CNN (TorchScript export): detection metrics on val, PyTorch vs ORT TF32 off / on."""
    ex, v = cfg.export, cfg.verify
    model, mcfg, class_names = load_detector(Path(ex.detectors.faster_rcnn))
    device = torch.device("cuda")
    model.to(device)
    recall_thr = float(mcfg.model.recall_score_thr)
    check_split("selection", "neu_det", "val")
    ds = neu_dataset(mcfg, "val")
    gts = ds.ground_truth(len(ds))
    loader = make_loader(ds, 1, shuffle=False, num_workers=2, seed=int(mcfg.seed))
    with strict_fp32():
        ref = faster_rcnn.predict(model, loader, device, amp=False)

    def ort_dets(tf32: bool) -> list[Detections]:
        sess = make_session(str(onnx_path), tf32=tf32)
        name = sess.get_inputs()[0].name
        out: list[Detections] = []
        for i in range(len(ds)):
            img, _, meta = ds[i]
            boxes, scores, labels = sess.run(None, {name: img.numpy().astype(np.float32)})
            arr = meta["affine"].invert(np.asarray(boxes, dtype=np.float64).reshape(-1, 4))
            h, w = meta["size"]
            out.append(Detections.of(clip_boxes(arr, w, h), np.asarray(scores), np.asarray(labels)))
        return out

    variants: dict[str, Any] = {
        REFERENCE: {"metrics": {k: detection_metrics(ref, gts, class_names, recall_thr)[k]
                                for k in KEYS},
                    "mean_detections": float(np.mean([len(d.scores) for d in ref]))}
    }
    for tf32 in v.tf32:
        dets = ort_dets(bool(tf32))
        variants[f"ort_tf32_{'on' if tf32 else 'off'}"] = {
            "metrics": {k: detection_metrics(dets, gts, class_names, recall_thr)[k] for k in KEYS},
            "mean_detections": float(np.mean([len(d.scores) for d in dets])),
        }
    return {"split": "val", "n_images": len(ds), "variants": variants}

