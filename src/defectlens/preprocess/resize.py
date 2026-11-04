"""Resize to a fixed square input, optionally letterboxed (aspect ratio kept, padded)."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from defectlens.preprocess.geometry import Affine

Image = npt.NDArray[np.uint8]


def resize_to_square(
    img: Image, size: int, letterbox: bool = True, pad_value: int = 114
) -> tuple[Image, Affine]:
    """Resize ``img`` (H, W[, C]) to ``size`` x ``size``.

    With ``letterbox`` the aspect ratio is kept and the image is centred on a ``pad_value``
    canvas, so a uniform :class:`Affine` describes the mapping. Without it the image is
    stretched; that is not uniform, so it is rejected for box work (use letterbox).
    Interpolation: INTER_AREA when shrinking, INTER_LINEAR when enlarging.
    """
    h, w = img.shape[:2]
    if not letterbox:
        msg = "stretch resize is non-uniform and would break box mapping; use letterbox=True"
        raise ValueError(msg)
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, *img.shape[2:]), pad_value, dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, Affine(s, float(left), float(top))
