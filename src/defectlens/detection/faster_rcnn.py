"""Faster R-CNN ResNet50-FPN v2 (torchvision): model, training epoch and prediction.

The COCO-pretrained weights are downloaded by torchvision on first use. Predictions are mapped
back to the original image frame so metrics are comparable across detectors.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from defectlens.detection.datamodule import Meta, Target
from defectlens.eval.detection import Detections
from defectlens.preprocess.geometry import clip_boxes


def build_model(cfg: Any, num_classes: int) -> FasterRCNN:
    """``num_classes`` excludes background. ``cfg.pretrained`` loads COCO weights first."""
    pretrained = bool(cfg.pretrained)
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,
        trainable_backbone_layers=int(cfg.trainable_backbone_layers) if pretrained else None,
        min_size=int(cfg.input_size),  # inputs are already letterboxed to this size
        max_size=int(cfg.input_size),
        box_score_thresh=float(cfg.score_thresh),
        box_detections_per_img=int(cfg.detections_per_img),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def make_optimizer(model: FasterRCNN, cfg: Any) -> SGD:
    params = [p for p in model.parameters() if p.requires_grad]
    return SGD(
        params, lr=float(cfg.lr), momentum=float(cfg.momentum), weight_decay=float(cfg.weight_decay)
    )


def make_scheduler(opt: SGD, total_iters: int, warmup_iters: int) -> LambdaLR:
    """Linear warm-up, then cosine decay to zero, stepped once per iteration."""

    def factor(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(opt, factor)


def _to_device(
    images: Sequence[torch.Tensor], targets: Sequence[Target], device: torch.device
) -> tuple[list[torch.Tensor], list[Target]]:
    return (
        [i.to(device, non_blocking=True) for i in images],
        [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets],
    )


def train_one_epoch(
    model: FasterRCNN,
    loader: Iterable[tuple[list[torch.Tensor], list[Target], list[Meta]]],
    opt: SGD,
    sched: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool,
    grad_clip: float,
    max_batches: int | None = None,
) -> float:
    """One pass over ``loader``; returns the mean total loss. Raises on a non-finite loss."""
    model.train()
    total, n = 0.0, 0
    for step, (images, targets, _) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images, targets = _to_device(images, targets, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = sum(model(images, targets).values())
        if not torch.isfinite(loss):
            msg = f"non-finite loss {loss.item()} at step {step}"
            raise FloatingPointError(msg)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def predict(
    model: FasterRCNN,
    loader: Iterable[tuple[list[torch.Tensor], list[Target], list[Meta]]],
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None = None,
) -> list[Detections]:
    """Detections per image, in dataset order, in the original image frame."""
    model.eval()
    out: list[Detections] = []
    for step, (images, _, metas) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            preds = model([i.to(device) for i in images])
        for p, meta in zip(preds, metas, strict=True):
            boxes = p["boxes"].float().cpu().numpy().astype(np.float64)
            boxes = meta["affine"].invert(boxes)
            h, w = meta["size"]
            out.append(
                Detections.of(
                    clip_boxes(boxes, w, h),
                    p["scores"].float().cpu().numpy(),
                    p["labels"].cpu().numpy(),
                )
            )
    return out
