"""Spike: load D-FINE-S (HF), run a forward pass on GPU, print shapes and peak VRAM."""

from __future__ import annotations

import sys

import torch
from transformers import DFineForObjectDetection

from defectlens.seed import seed_everything

MODEL_ID = "ustc-community/dfine-small-coco"
SIZE = 640


def main() -> int:
    seed_everything(42)
    model = DFineForObjectDetection.from_pretrained(MODEL_ID).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model {MODEL_ID}  params {n_params:.1f}M")
    x = torch.randn(1, 3, SIZE, SIZE, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model(pixel_values=x)
    torch.cuda.synchronize()
    print(f"logits    {tuple(out.logits.shape)}")
    print(f"pred_boxes {tuple(out.pred_boxes.shape)}")
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"peak VRAM (fp16 inference, batch 1) {peak:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
