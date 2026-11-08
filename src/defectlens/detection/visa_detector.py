"""A Faster R-CNN trained on VisA images (colour, with normal images as negatives), Phase 4b.

Reuses the model, optimiser, scheduler and epoch loop of ``detection.faster_rcnn``. There is no
validation split and no checkpoint selection: the epoch count is fixed in the config before the
run, and the last epoch is used. Everything a detector says about an image is reduced to one
number, the highest box score (0 when nothing is detected), so it can be thresholded and ranked
exactly like an anomaly score.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torchvision.models.detection import FasterRCNN

from defectlens.data.coco import CocoImage
from defectlens.detection import faster_rcnn
from defectlens.detection.common import make_scheduler
from defectlens.detection.datamodule import NeuDetDataset, build_augment, make_loader
from defectlens.preprocess.pipeline import PreprocessConfig, Preprocessor


class VisaDetDataset(NeuDetDataset):
    """Colour VisA images through the shared preprocessing (letterboxed to ``input_size``).

    Image-level negatives (no boxes) are augmented like the others (see ``augment_empty``).
    """

    augment_empty = True

    def read_image(self, it: CocoImage) -> npt.NDArray[np.uint8]:
        bgr = cv2.imread(str(self.images_dir / it.file_name), cv2.IMREAD_COLOR)
        if bgr is None:
            msg = f"cannot read {self.images_dir / it.file_name}"
            raise FileNotFoundError(msg)
        return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def make_dataset(
    coco_path: Path, root: Path, model_cfg: Any, preprocess_cfg: Any, *, augment_seed: int | None
) -> VisaDetDataset:
    """Dataset over a COCO file; ``augment_seed=None`` means no augmentation (scoring)."""
    pre_cfg = {**preprocess_cfg, "size": int(model_cfg.input_size)}
    pre = Preprocessor(PreprocessConfig.from_dict(pre_cfg))
    aug = None if augment_seed is None else build_augment(model_cfg.augment, augment_seed)
    return VisaDetDataset(coco_path, root, pre, aug)


@dataclass
class TrainStats:
    """What a training run measured about itself."""

    epochs: int = 0
    iters_per_epoch: int = 0
    seconds: float = 0.0
    peak_vram_alloc_mib: float = 0.0
    peak_vram_reserved_mib: float = 0.0
    loss: list[float] = field(default_factory=list)


def train_detector(
    model_cfg: Any,
    train_ds: NeuDetDataset,
    *,
    device: torch.device,
    amp: bool,
    seed: int,
    num_workers: int,
    epochs: int,
    max_batches: int | None = None,
    on_epoch: Callable[[int, float], None] | None = None,
) -> tuple[FasterRCNN, TrainStats]:
    """Train from the (COCO-pretrained) start for exactly ``epochs`` epochs; no model selection.

    ``max_batches`` caps the batches per epoch (smoke runs). ``on_epoch(epoch, mean_loss)`` is
    called after each epoch (MLflow logging).
    """
    amp = amp and device.type == "cuda"
    loader = make_loader(
        train_ds, int(model_cfg.batch_size), shuffle=True, num_workers=num_workers, seed=seed
    )
    iters = min(len(loader), max_batches) if max_batches else len(loader)
    model = faster_rcnn.build_model(model_cfg, len(train_ds.class_names)).to(device)
    opt = faster_rcnn.make_optimizer(model, model_cfg)
    sched = make_scheduler(opt, epochs * iters, int(model_cfg.warmup_iters))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    stats = TrainStats(epochs=epochs, iters_per_epoch=iters)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        loss = faster_rcnn.train_one_epoch(
            model, loader, opt, sched, scaler, device, amp=amp,
            grad_clip=float(model_cfg.grad_clip), max_batches=max_batches,
        )
        stats.loss.append(loss)
        if on_epoch is not None:
            on_epoch(epoch, loss)
    stats.seconds = time.perf_counter() - t0
    if device.type == "cuda":
        stats.peak_vram_alloc_mib = torch.cuda.max_memory_allocated() / 2**20
        stats.peak_vram_reserved_mib = torch.cuda.max_memory_reserved() / 2**20
    return model, stats


def image_scores(
    model: FasterRCNN,
    ds: NeuDetDataset,
    *,
    device: torch.device,
    amp: bool,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    """``{file_name: highest box score}`` for every image of ``ds`` (0.0 if nothing is detected)."""
    loader = make_loader(ds, batch_size, shuffle=False, num_workers=num_workers, seed=seed)
    dets = faster_rcnn.predict(
        model, loader, device, amp=amp and device.type == "cuda", max_batches=max_batches
    )
    return {
        it.file_name: float(d.scores.max()) if len(d.scores) else 0.0
        for it, d in zip(ds.items, dets, strict=False)
    }
