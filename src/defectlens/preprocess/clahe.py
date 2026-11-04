"""CLAHE contrast enhancement and channel handling."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]


def apply_clahe(img: Image, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> Image:
    """CLAHE on a uint8 image. Grayscale is enhanced directly; 3-channel BGR is enhanced on the
    L channel of LAB so colours are not shifted. Deterministic: no random state involved."""
    if img.dtype != np.uint8:
        msg = f"CLAHE expects uint8, got {img.dtype}"
        raise ValueError(msg)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    if img.ndim == 2:
        return np.asarray(clahe.apply(img), dtype=np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.asarray(clahe.apply(img[:, :, 0]), dtype=np.uint8)[:, :, None]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return np.asarray(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), dtype=np.uint8)


def to_three_channel(img: Image) -> Image:
    """Return (H, W, 3): grayscale (H, W) or (H, W, 1) is replicated, BGR passes through."""
    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.shape[2] == 3:
        return img
    msg = f"unsupported channel count {img.shape[2]}"
    raise ValueError(msg)
