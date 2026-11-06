"""NEU-DET as a torchvision-style detection dataset, built on the shared preprocessing path.

Each item is the letterboxed image (3 channels, ``[0, 1]``), a target dict with ``boxes`` (xyxy,
in the letterboxed frame) and ``labels`` (COCO category ids 1..K, so 0 stays background), and the
affine that maps predictions back to the original 200x200 frame, where every detector is scored.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from defectlens.data.coco import CocoImage, read_coco
from defectlens.eval.detection import GroundTruth
from defectlens.preprocess.pipeline import Preprocessor

Target = dict[str, torch.Tensor]
Meta = dict[str, Any]
Item = tuple[torch.Tensor, Target, Meta]


def build_augment(cfg: Any, seed: int) -> A.Compose:
    """Flips, brightness/contrast jitter, optional scale/shift (no preferred steel orientation)."""
    ops: list[A.BasicTransform] = [
        A.HorizontalFlip(p=float(cfg.hflip)),
        A.VerticalFlip(p=float(cfg.vflip)),
        A.RandomBrightnessContrast(0.2, 0.2, p=float(cfg.brightness_contrast)),
    ]
    p_affine = float(getattr(cfg, "scale_translate", 0.0))  # optional: absent for Faster R-CNN
    if p_affine > 0:
        # Scale 0.75-1.25 and shift up to 10%; borders filled with mid-gray, boxes follow.
        ops.append(A.Affine(scale=(0.75, 1.25), translate_percent=(-0.1, 0.1), rotate=0,
                            border_mode=cv2.BORDER_CONSTANT, fill=114, p=p_affine))
    return A.Compose(
        ops,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.5),
        seed=seed,
    )


class NeuDetDataset(Dataset[Item]):
    def __init__(
        self,
        coco_path: Path,
        images_dir: Path,
        preprocessor: Preprocessor,
        augment: A.Compose | None = None,
    ) -> None:
        self.items, self.class_names = read_coco(coco_path)
        self.images_dir = images_dir
        self.pre = preprocessor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def ground_truth(self, n: int | None = None) -> list[GroundTruth]:
        """Ground truth in the original frame, in dataset order (first ``n`` images)."""
        return [GroundTruth.of(it.boxes, it.labels) for it in self.items[:n]]

    def __getitem__(self, idx: int) -> Item:
        it: CocoImage = self.items[idx]
        gray = cv2.imread(str(self.images_dir / it.file_name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            msg = f"cannot read {self.images_dir / it.file_name}"
            raise FileNotFoundError(msg)
        res = self.pre(np.asarray(gray, dtype=np.uint8), it.boxes)
        img, boxes, labels = res.image, res.boxes, it.labels[res.keep]
        if self.augment is not None and len(boxes):
            h, w = img.shape[:2]
            out = self.augment(
                image=img,
                bboxes=np.clip(boxes, 0, [w, h, w, h]).tolist(),
                labels=labels.tolist(),
            )
            img = out["image"]
            boxes = np.asarray(out["bboxes"], dtype=np.float64).reshape(-1, 4)
            labels = np.asarray(out["labels"], dtype=np.int64)
        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        meta = {"image_id": it.image_id, "affine": res.affine, "size": (it.height, it.width)}
        return tensor, target, meta


def collate(batch: Sequence[Item]) -> tuple[list[torch.Tensor], list[Target], list[Meta]]:
    images, targets, metas = zip(*batch, strict=True)
    return list(images), list(targets), list(metas)


def _seed_worker(_: int) -> None:
    """Give each worker (and each epoch) its own augmentation stream, derived from the base seed."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        ds = info.dataset
        if isinstance(ds, NeuDetDataset) and ds.augment is not None:
            ds.augment.set_random_seed(torch.initial_seed() % 2**32)


def make_loader(
    ds: NeuDetDataset, batch_size: int, *, shuffle: bool, num_workers: int, seed: int
) -> DataLoader[Item]:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

