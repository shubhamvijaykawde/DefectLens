"""One preprocessing path for training, evaluation and serving.

Order: ROI crop -> CLAHE (at native resolution) -> letterbox resize -> 3 channels.
Geometry is tracked as a single :class:`Affine` from original to output coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from defectlens.preprocess.clahe import apply_clahe, to_three_channel
from defectlens.preprocess.geometry import Affine, Boxes, as_boxes
from defectlens.preprocess.resize import resize_to_square
from defectlens.preprocess.roi import crop_roi

Image = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class PreprocessConfig:
    """Mirrors ``configs/preprocess/default.yaml``."""

    size: int = 640
    letterbox: bool = True
    pad_value: int = 114
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    roi: tuple[int, int, int, int] | None = None
    roi_min_visible: float = 0.5
    three_channel: bool = True

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PreprocessConfig:
        """Build from a plain dict (e.g. ``OmegaConf.to_container``), nested ``clahe``/``roi``."""
        clahe = d.get("clahe") or {}
        roi = d.get("roi")
        grid = clahe.get("tile_grid", (8, 8))
        return cls(
            size=int(d.get("size", 640)),
            letterbox=bool(d.get("letterbox", True)),
            pad_value=int(d.get("pad_value", 114)),
            clahe=bool(clahe.get("enabled", False)),
            clahe_clip_limit=float(clahe.get("clip_limit", 2.0)),
            clahe_tile_grid=(int(grid[0]), int(grid[1])),
            roi=(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])) if roi else None,
            roi_min_visible=float(d.get("roi_min_visible", 0.5)),
            three_channel=bool(d.get("three_channel", True)),
        )


@dataclass(frozen=True)
class PreprocessResult:
    image: Image
    boxes: Boxes  # in output coordinates, only boxes kept by the ROI crop
    keep: npt.NDArray[np.bool_]  # mask over the input boxes
    affine: Affine  # original -> output coordinates
    original_size: tuple[int, int] = field(default=(0, 0))  # (H, W)

    def to_original(self, boxes: Boxes) -> Boxes:
        """Map boxes predicted in output coordinates back to the original image."""
        return self.affine.invert(boxes)


class Preprocessor:
    """Stateless callable; the same object is used by training code and the API."""

    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg

    def __call__(self, img: Image, boxes: npt.ArrayLike | None = None) -> PreprocessResult:
        cfg = self.cfg
        b = as_boxes(boxes)
        h, w = img.shape[:2]
        keep = np.ones(len(b), dtype=bool)
        aff = Affine()
        if cfg.roi is not None:
            img, b, keep, aff = crop_roi(img, b, cfg.roi, cfg.roi_min_visible)
        else:
            b = b.copy()
        if cfg.clahe:
            img = apply_clahe(img, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
        img, rs = resize_to_square(img, cfg.size, cfg.letterbox, cfg.pad_value)
        aff = aff.then(rs)
        b = rs.apply(b)
        if cfg.three_channel:
            img = to_three_channel(img)
        return PreprocessResult(img, b, keep, aff, (h, w))
