"""VisA image sets built from our fixed split files (not anomalib's own splits).

``train`` is normal images only; ``threshold_calibration`` and ``threshold_evaluation`` are the two
disjoint halves of the official test set (see ``defectlens.data.visa_splits``). Images are
resized to a square ``image_size`` (the anomaly models use their own input size, not the NEU-DET
preprocessing pipeline) and normalised with ImageNet statistics.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

from defectlens.data.visa_splits import read_one_cls

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "threshold_calibration", "threshold_evaluation")


def load_split_paths(splits_dir: Path, category: str, split: str) -> list[str]:
    """Image paths (relative to the VisA root) of one split of one category."""
    if split not in SPLITS:
        msg = f"unknown split {split!r}, expected one of {SPLITS}"
        raise ValueError(msg)
    data = json.loads((splits_dir / f"visa_{category}.json").read_text())
    return list(data[split])


def few_shot_subset(paths: Sequence[str], shots: int | None, seed: int) -> list[str]:
    """A fixed random ``shots``-image subset (sorted), or all paths when ``shots`` is None."""
    ordered = sorted(paths)
    if shots is None or shots >= len(ordered):
        return ordered
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ordered, size=shots, replace=False).tolist())


def denormalize(images: torch.Tensor) -> npt.NDArray[np.float32]:
    """(B, 3, H, W) normalised tensor -> (B, H, W, 3) float array in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    out = (images * std + mean).clamp(0, 1).permute(0, 2, 3, 1)
    return out.cpu().numpy().astype(np.float32)


class VisaImages(Dataset[dict[str, Any]]):
    """Images of one VisA split. Items: ``image`` (3,H,W) float, ``mask`` (H,W) float 0/1,
    ``label`` 0 normal / 1 anomaly, ``path``."""

    def __init__(
        self,
        root: Path,
        csv_path: Path,
        category: str,
        paths: Sequence[str],
        image_size: tuple[int, int],
    ) -> None:
        rows = {r["image"]: r for r in read_one_cls(csv_path, category)}
        missing = [p for p in paths if p not in rows]
        if missing:
            msg = f"{len(missing)} paths are not in {csv_path} for {category}, e.g. {missing[0]}"
            raise ValueError(msg)
        self.root = root
        self.image_size = (int(image_size[0]), int(image_size[1]))  # (height, width)
        self.paths = list(paths)
        self.labels = [int(rows[p]["label"] == "anomaly") for p in self.paths]
        self.mask_paths = [rows[p]["mask"] for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        h, w = self.image_size
        bgr = cv2.imread(str(self.root / self.paths[index]), cv2.IMREAD_COLOR)
        if bgr is None:
            msg = f"cannot read image {self.root / self.paths[index]}"
            raise FileNotFoundError(msg)
        rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
            IMAGENET_STD, dtype=np.float32
        )
        mask = np.zeros((h, w), dtype=np.float32)
        if self.labels[index] and self.mask_paths[index]:
            m = cv2.imread(str(self.root / self.mask_paths[index]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                msg = f"cannot read mask {self.root / self.mask_paths[index]}"
                raise FileNotFoundError(msg)
            mask = (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        return {
            "image": torch.from_numpy(x).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(mask),
            "label": self.labels[index],
            "path": self.paths[index],
        }


def make_loader(
    dataset: VisaImages, batch_size: int, num_workers: int = 2, shuffle: bool = False
) -> DataLoader[dict[str, Any]]:
    """Plain DataLoader; order is deterministic unless ``shuffle``."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False
    )
