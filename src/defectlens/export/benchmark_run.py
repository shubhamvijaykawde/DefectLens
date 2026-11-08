"""Measure one (model, variant) in this process: used by ``scripts/benchmark.py`` as a child.

See ``export/benchmark.py`` for the protocol. Inputs are the first image of a selection split
(NEU-DET ``val`` or VisA ``threshold_calibration``); nothing from an evaluation split is read.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from defectlens.anomaly.datamodule import VisaImages, load_split_paths
from defectlens.eval.scoring import load_anomaly_model
from defectlens.eval.thresholds import check_split
from defectlens.export.benchmark import (
    MemorySampler,
    Variant,
    nvidia_smi_used_mib,
    summarize,
    time_calls,
)
from defectlens.export.onnx_export import load_detector
from defectlens.export.verify import make_session, strict_fp32
from defectlens.export.verify_run import neu_dataset

DETECTORS = ("dfine_s", "faster_rcnn")


def _device_used_mib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**20


def prepare(
    cfg: DictConfig, model: str, category: str | None, variant: Variant
) -> tuple[Callable[[], object], dict[str, Any]]:
    """The timed call for one model and variant, and facts about the setup."""
    onnx_dir = Path(cfg.export.out_dir)
    cuda = variant.device == "cuda"
    info: dict[str, Any] = {}
    # --- input and, for PyTorch, the model --------------------------------------------------
    if model in DETECTORS:
        path = cfg.export.detectors[model]
        net, mcfg, _ = load_detector(Path(path))
        check_split("selection", "neu_det", "val")
        img = neu_dataset(mcfg, "val")[0][0]  # (3, S, S) in [0, 1]
        x = img.unsqueeze(0) if model == "dfine_s" else img
        onnx_path = onnx_dir / f"{model}.onnx"
    else:
        assert category is not None
        obj, mcfg = load_anomaly_model(model, category, "cuda" if cuda else "cpu")
        data = mcfg.data
        size = (int(mcfg.model.image_size[0]), int(mcfg.model.image_size[1]))
        check_split("selection", "visa", "threshold_calibration")
        paths = load_split_paths(Path(data.splits_dir), category, "threshold_calibration")
        ds = VisaImages(Path(data.raw_dir), Path(data.csv), category, paths[:1], size)
        x = ds[0]["image"].unsqueeze(0)
        onnx_path = onnx_dir / f"{model}_{category}.onnx"
    info["input_shape"] = list(x.shape)

    # --- the callable -----------------------------------------------------------------------
    if variant.kind == "ort":
        t0 = time.perf_counter()
        sess = make_session(str(onnx_path), tf32=variant.tf32, device=variant.device)
        info["load_seconds"] = time.perf_counter() - t0
        info["providers"] = sess.get_providers()
        name = sess.get_inputs()[0].name
        arr = np.ascontiguousarray(x.numpy().astype(np.float32))

        def run_ort() -> object:
            return sess.run(None, {name: arr})

        return run_ort, info

    if variant.precision not in ("fp32", "fp32_strict", "fp16_amp"):
        msg = f"unknown PyTorch precision {variant.precision!r}"
        raise ValueError(msg)
    dev = torch.device("cuda")
    xt = x.to(dev)
    amp = variant.precision == "fp16_amp"
    strict = variant.precision == "fp32_strict"
    if model in DETECTORS:
        net.to(dev)

        def forward() -> list[torch.Tensor]:
            if model == "dfine_s":
                out = net(pixel_values=xt)
                return [out.logits, out.pred_boxes]
            pred = net([xt])[0]
            return [pred["boxes"], pred["scores"], pred["labels"]]
    else:
        def forward() -> list[torch.Tensor]:
            s, m = obj.predict(xt)
            return [s, m]

    def run_torch() -> object:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            if strict:
                with strict_fp32():
                    outs = forward()
            else:
                outs = forward()
        return [o.float().cpu() for o in outs]  # the copy back to the host also synchronises

    info["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32) and not strict
    return run_torch, info


def measure(cfg: DictConfig, model: str, category: str | None, label: str) -> dict[str, Any]:
    """One result row: latency summary, load time, peak RAM and device memory."""
    b = cfg.benchmark
    variant = Variant.from_label(label)
    baseline = nvidia_smi_used_mib()  # before this process touches the GPU
    if variant.device == "cuda":
        torch.zeros(1, device="cuda")
    with MemorySampler(_device_used_mib, float(b.gpu_poll_ms) / 1000.0) as sampler:
        call, info = prepare(cfg, model, category, variant)
        times = time_calls(call, warmup=int(b.warmup), runs=int(b.runs))
    return {
        "model": model, "category": category, "variant": label, "batch": int(b.batch),
        "warmup": int(b.warmup), "runs": int(b.runs), "latency": summarize(times),
        "peak_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "device_baseline_mib": baseline,
        "device_peak_mib": sampler.peak if variant.device == "cuda" else None,
        "device_used_by_variant_mib": (
            sampler.peak - baseline if variant.device == "cuda" and baseline is not None else None
        ),
        **info,
    }
