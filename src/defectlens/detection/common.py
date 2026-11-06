"""Pieces shared by every detector trainer: LR schedule and device transfer."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from defectlens.detection.datamodule import Target


def make_scheduler(opt: Optimizer, total_iters: int, warmup_iters: int) -> LambdaLR:
    """Linear warm-up, then cosine decay to zero, stepped once per optimizer step."""

    def factor(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(opt, factor)


def to_device(
    images: Sequence[torch.Tensor], targets: Sequence[Target], device: torch.device
) -> tuple[list[torch.Tensor], list[Target]]:
    return (
        [i.to(device, non_blocking=True) for i in images],
        [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets],
    )
