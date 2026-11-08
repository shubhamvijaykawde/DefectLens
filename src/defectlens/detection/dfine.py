"""D-FINE (Hugging Face Transformers): model, training epoch with accumulation, and prediction.

Fine-tunes the COCO-pretrained checkpoint ``cfg.hf_id``; the class heads are re-initialised for the
NEU-DET classes. Our dataset gives xyxy boxes in pixels with labels 1..K; D-FINE wants normalised
cxcywh boxes with labels 0..K-1, and its outputs are per-query sigmoid scores, so conversion
happens at the two edges (``to_hf_labels`` and ``predict``). Predictions are mapped back to the
original image frame like every other detector's.
"""

from __future__ import annotations

from collections.abc import Sized
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import DFineConfig, DFineForObjectDetection

from defectlens.detection.common import to_device
from defectlens.detection.datamodule import Item, Target
from defectlens.eval.detection import Detections
from defectlens.preprocess.geometry import clip_boxes

BACKBONE_PREFIX = "model.backbone."
MIN_BOX = 1e-4  # normalised; boxes thinner than this would give a NaN GIoU


def tiny_config(num_classes: int) -> DFineConfig:
    """Small random-weight architecture for tests and the CPU smoke path (no download)."""
    backbone = {
        "depths": [3, 4, 6, 3], "downsample_in_bottleneck": False,
        "downsample_in_first_stage": False, "embedding_size": 32, "hidden_act": "relu",
        "hidden_sizes": [128, 256, 512, 1024], "layer_type": "basic", "model_type": "hgnet_v2",
        "num_channels": 3, "out_features": ["stage2", "stage3", "stage4"],
        "out_indices": [2, 3, 4], "stage_downsample": [False, True, True, True],
        "stage_in_channels": [16, 64, 256, 512], "stage_kernel_size": [3, 3, 5, 5],
        "stage_light_block": [False, False, True, True], "stage_mid_channels": [16, 32, 64, 128],
        "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4"],
        "stage_num_blocks": [1, 1, 2, 1], "stage_numb_of_layers": [3, 3, 3, 3],
        "stage_out_channels": [64, 256, 512, 1024], "stem_channels": [3, 16, 16],
        "use_learnable_affine_block": True,
    }
    return DFineConfig(  # type: ignore[call-arg]  # num_labels etc. come via PretrainedConfig kwargs
        backbone_config=backbone, num_labels=num_classes, num_queries=20, num_denoising=10,
        decoder_layers=2, d_model=64, encoder_hidden_dim=64, decoder_in_channels=[64, 64, 64],
        encoder_in_channels=[128, 256, 512], encoder_ffn_dim=128, decoder_ffn_dim=128,
        encoder_attention_heads=2, decoder_attention_heads=2, max_num_bins=8,
        use_pretrained_backbone=False,
    )


def build_model(cfg: Any, num_classes: int) -> DFineForObjectDetection:
    """``num_classes`` excludes background (D-FINE has none). ``pretrained=False`` builds the
    tiny random-weight test architecture, not a from-scratch D-FINE-S."""
    if bool(cfg.pretrained):
        model = DFineForObjectDetection.from_pretrained(
            str(cfg.hf_id), num_labels=num_classes, ignore_mismatched_sizes=True
        )
        assert isinstance(model, DFineForObjectDetection)
        return model
    return DFineForObjectDetection(tiny_config(num_classes))


def make_optimizer(model: torch.nn.Module, cfg: Any) -> AdamW:
    """AdamW with a lower backbone LR and no weight decay on biases and norm parameters."""
    groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups.setdefault((name.startswith(BACKBONE_PREFIX), p.ndim <= 1), []).append(p)
    lr, wd = float(cfg.lr), float(cfg.weight_decay)
    param_groups = [
        {"params": ps, "lr": lr * (float(cfg.backbone_lr_mult) if backbone else 1.0),
         "weight_decay": 0.0 if no_decay else wd}
        for (backbone, no_decay), ps in sorted(groups.items())
    ]
    return AdamW(param_groups, lr=lr, betas=(0.9, 0.999))


def to_hf_labels(targets: list[Target], size: tuple[int, int]) -> list[dict[str, torch.Tensor]]:
    """xyxy pixels + labels 1..K -> normalised cxcywh + labels 0..K-1. ``size`` is (H, W)."""
    h, w = size
    out = []
    for t in targets:
        b = t["boxes"].clone().float()
        b[:, [0, 2]] = (b[:, [0, 2]] / w).clamp(0.0, 1.0)
        b[:, [1, 3]] = (b[:, [1, 3]] / h).clamp(0.0, 1.0)
        labels = t["labels"] - 1
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        cxcywh = torch.stack([cx, cy, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], dim=1)
        ok = (cxcywh[:, 2] > MIN_BOX) & (cxcywh[:, 3] > MIN_BOX)
        out.append({"class_labels": labels[ok], "boxes": cxcywh[ok]})
    return out


def train_one_epoch(
    model: DFineForObjectDetection,
    loader: DataLoader[Item],
    opt: AdamW,
    sched: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool,
    grad_clip: float,
    max_batches: int | None = None,
    accum_steps: int = 1,
) -> float:
    """One pass over ``loader`` with gradient accumulation; returns the mean total loss.

    The optimizer and scheduler step every ``accum_steps`` batches and once more for a final
    partial group. Raises on a non-finite loss.
    """
    model.train()
    n_batches = len(loader) if isinstance(loader, Sized) else 10**9
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)
    total, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for step, (images, targets, _) in enumerate(loader):
        if step >= n_batches:
            break
        images, targets = to_device(images, targets, device)
        x = torch.stack(images)
        labels = to_hf_labels(targets, (x.shape[-2], x.shape[-1]))
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = model(pixel_values=x, labels=labels).loss
        if loss is None or not torch.isfinite(loss):
            msg = f"non-finite loss {None if loss is None else loss.item()} at step {step}"
            raise FloatingPointError(msg)
        scaler.scale(loss / accum_steps).backward()
        total += float(loss.item())
        n += 1
        if (step + 1) % accum_steps == 0 or step + 1 == n_batches:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
    return total / max(n, 1)


def decode(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    size_hw: tuple[int, int],
    *,
    score_thresh: float = 0.001,
    detections_per_img: int = 100,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Raw network outputs -> per image ``(xyxy pixels, scores, labels 1..K)`` in the input frame.

    Takes the top ``detections_per_img`` (query, class) pairs by sigmoid score and drops scores
    below ``score_thresh``. Shared by the PyTorch path (``predict``) and the ONNX Runtime path
    (``export/verify.py``), so a difference between the two can only come from the network.
    """
    probs = logits.float().sigmoid()
    _, n_q, n_c = probs.shape
    scores, idx = probs.flatten(1).topk(min(detections_per_img, n_q * n_c), dim=1)
    query, cls = idx // n_c, idx % n_c
    cxcywh = pred_boxes.float().gather(1, query.unsqueeze(-1).expand(-1, -1, 4))
    h, w = size_hw
    xyxy = torch.cat([cxcywh[..., :2] - cxcywh[..., 2:] / 2,
                      cxcywh[..., :2] + cxcywh[..., 2:] / 2], dim=-1)
    xyxy = xyxy * torch.tensor([w, h, w, h], device=xyxy.device, dtype=xyxy.dtype)
    out = []
    for i in range(probs.shape[0]):
        keep = scores[i] >= score_thresh
        out.append((xyxy[i][keep], scores[i][keep], cls[i][keep] + 1))
    return out


def to_detections(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, meta: dict[str, Any]
) -> Detections:
    """Map one image's decoded boxes back to the original image frame and clip them."""
    arr = meta["affine"].invert(boxes.cpu().numpy().astype(np.float64))
    oh, ow = meta["size"]
    return Detections.of(
        clip_boxes(arr, ow, oh), scores.cpu().numpy(), labels.cpu().numpy()
    )


@torch.no_grad()
def predict(
    model: DFineForObjectDetection,
    loader: DataLoader[Item],
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None = None,
    score_thresh: float = 0.001,
    detections_per_img: int = 100,
) -> list[Detections]:
    """Top-``detections_per_img`` (query, class) pairs per image, in the original image frame."""
    model.eval()  # type: ignore[no-untyped-call]
    out: list[Detections] = []
    for step, (images, _, metas) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        x = torch.stack([i.to(device) for i in images])
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            res = model(pixel_values=x)
        decoded = decode(
            res.logits, res.pred_boxes, (x.shape[-2], x.shape[-1]),
            score_thresh=score_thresh, detections_per_img=detections_per_img,
        )
        out += [to_detections(*d, meta) for d, meta in zip(decoded, metas, strict=True)]
    return out
