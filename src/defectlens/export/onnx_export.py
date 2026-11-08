"""ONNX export of the five trained models (Phase 5a), with failures recorded, not hidden.

Every export goes through ``export_module`` on CPU in float32 with the Dynamo exporter
(``optimize=False``: with the exporter's own optimization on, ORT 1.31.1 returned wrong outputs for
the D-FINE stem in the Phase 0c spike). The batch size is fixed at 1, which is what the API
serves. A model that can not be exported yields an ``ExportRecord`` with ``status="failed"`` and
the error text, so the report says what happened instead of leaving a gap.

ONNX input / output contract (also written to ``<file>.json`` next to each ONNX file):

* ``dfine_s``: ``pixel_values`` (1,3,S,S) float32 in [0,1] (letterboxed, see ``preprocess``) ->
  ``logits`` (1,300,K), ``boxes`` (1,300,4) normalised cxcywh. Top-k and thresholding stay outside
  (``detection.dfine.decode``), like in the PyTorch path.
* ``faster_rcnn``: ``image`` (3,S,S) float32 in [0,1] -> ``boxes`` (N,4) xyxy, ``scores`` (N,),
  ``labels`` (N,) (torchvision's own post-processing is part of the graph).
* ``patchcore`` / ``anomaly_dino`` / ``efficientad``: ``image`` (1,3,H,W) ImageNet-normalised, as
  ``VisaImages`` produces it -> ``score`` (1,), ``anomaly_map`` (1,H,W).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import onnx
import torch
from omegaconf import OmegaConf

from defectlens.data.coco import read_coco
from defectlens.detection.trainer import BACKENDS
from defectlens.preprocess.anomaly_input import IMAGENET_MEAN, IMAGENET_STD

# transformers documents this switch for export and production: it disables shape sanity checks
# (torch_compilable_check) that the Dynamo translator can not convert (aten._is_all_true).
TRANSFORMERS_ENV = "TRANSFORMERS_DISABLE_TORCH_CHECK"


@dataclass
class ExportRecord:
    """What happened to one export; written to ``reports/onnx_export.json``."""

    model: str
    category: str | None
    status: str  # "ok" or "failed"
    path: str | None = None
    size_mib: float | None = None
    opset: dict[str, int] = field(default_factory=dict)
    seconds: float | None = None
    exporter: str = "dynamo"
    input_name: str = ""
    input_shape: list[int] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    reason: str = ""  # first lines of the error for a failed export

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------------
# Wrappers: plain tensors in, plain tensors out


class DFineWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


class FasterRCNNWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.model([image])[0]
        return pred["boxes"], pred["scores"], pred["labels"]


class AnomalyWrapper(torch.nn.Module):
    """Inference output of an anomalib torch model as ``(score (B,), map (B,H,W))``.

    ``unit_input`` marks EfficientAD, whose network normalises internally and so takes [0, 1]
    images: the ImageNet normalisation of the dataloader is undone inside the graph.
    """

    imagenet_mean: torch.Tensor
    imagenet_std: torch.Tensor

    def __init__(self, model: torch.nn.Module, *, unit_input: bool = False) -> None:
        super().__init__()
        self.model = model
        self.unit_input = unit_input
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = image * self.imagenet_std + self.imagenet_mean if self.unit_input else image
        out = self.model(x)
        return out.pred_score.reshape(-1), out.anomaly_map.squeeze(1)


# --------------------------------------------------------------------------------------------
# Loading the trained models on CPU (the same restore paths as Phase 4a)


def load_detector(checkpoint: Path) -> tuple[torch.nn.Module, Any, dict[int, str]]:
    """A trained NEU-DET detector in eval mode on CPU, its training config and its class names."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    _, class_names = read_coco(Path(cfg.data.processed_dir) / "val.json")
    model = BACKENDS[str(cfg.model.name)].build(cfg.model, len(class_names))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, class_names


def load_anomaly(model_name: str, category: str, device: str = "cpu") -> tuple[Any, Any]:
    """The fitted Phase 3 model of ``(model_name, category)``; returns ``(wrapper object, cfg)``."""
    from defectlens.eval.scoring import load_anomaly_model  # noqa: PLC0415  (heavy imports)

    return load_anomaly_model(model_name, category, device)


# --------------------------------------------------------------------------------------------
# Export


def _reason(exc: BaseException) -> str:
    """The exception chain as ``Type: first line`` pieces, ANSI colours removed."""
    parts: list[str] = []
    cur: BaseException | None = exc
    while cur is not None and len(parts) < 4:
        text = re.sub(r"\x1b\[[0-9;]*m", "", str(cur)).strip().splitlines()
        parts.append(f"{type(cur).__name__}: {text[0][:200] if text else ''}")
        cur = cur.__cause__
    return " <- ".join(parts)


def export_module(
    module: torch.nn.Module,
    example: Sequence[torch.Tensor],
    out_path: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamo: bool = True,
) -> ExportRecord:
    """Export on CPU, check the file and return its record. Never raises for an export problem."""
    rec = ExportRecord(
        model=out_path.stem, category=None, status="failed",
        exporter="dynamo" if dynamo else "torchscript",
        input_name=input_names[0], input_shape=list(example[0].shape),
        output_names=list(output_names),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(TRANSFORMERS_ENV, "1")
    t0 = time.perf_counter()
    try:
        module = module.eval().cpu()
        with torch.no_grad():
            if dynamo:
                prog = torch.onnx.export(
                    module, tuple(example), dynamo=True, optimize=False,
                    input_names=list(input_names), output_names=list(output_names),
                )
                assert prog is not None
                prog.save(str(out_path))
            else:
                torch.onnx.export(
                    module, tuple(example), str(out_path), dynamo=False,
                    input_names=list(input_names), output_names=list(output_names),
                )
        onnx.checker.check_model(str(out_path))  # by path: works above the 2 GB protobuf limit
        meta = onnx.load(str(out_path), load_external_data=False)
        rec.opset = {o.domain or "ai.onnx": int(o.version) for o in meta.opset_import}
        rec.status, rec.path = "ok", str(out_path)
        rec.size_mib = out_path.stat().st_size / 2**20
    except Exception as exc:  # the point of this function is to record, not to stop
        rec.reason = _reason(exc)
        out_path.unlink(missing_ok=True)
    rec.seconds = time.perf_counter() - t0
    if rec.status == "ok":
        sidecar = out_path.with_suffix(".json")
        sidecar.write_text(json.dumps(rec.to_dict(), indent=1))
    return rec


def export_dfine(checkpoint: Path, out_path: Path) -> ExportRecord:
    model, cfg, _ = load_detector(checkpoint)
    size = int(cfg.model.input_size)
    rec = export_module(
        DFineWrapper(model), (torch.rand(1, 3, size, size),), out_path,
        input_names=["pixel_values"], output_names=["logits", "boxes"],
    )
    rec.model = "dfine_s"
    return rec


def export_faster_rcnn(checkpoint: Path, out_path: Path, *, dynamo: bool) -> ExportRecord:
    model, cfg, _ = load_detector(checkpoint)
    size = int(cfg.model.input_size)
    rec = export_module(
        FasterRCNNWrapper(model), (torch.rand(3, size, size),), out_path,
        input_names=["image"], output_names=["boxes", "scores", "labels"], dynamo=dynamo,
    )
    rec.model = "faster_rcnn"
    return rec


def export_anomaly(model_name: str, category: str, out_path: Path) -> ExportRecord:
    obj, cfg = load_anomaly(model_name, category)
    h, w = (int(cfg.model.image_size[0]), int(cfg.model.image_size[1]))
    x = torch.randn(1, 3, h, w)
    wrapper = AnomalyWrapper(obj.model, unit_input=model_name == "efficientad")
    rec = export_module(
        wrapper, (x,), out_path, input_names=["image"], output_names=["score", "anomaly_map"]
    )
    rec.model, rec.category = model_name, category
    return rec


Exporter = Callable[[], ExportRecord]
