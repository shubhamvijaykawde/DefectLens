"""Fixed region-of-interest crop with box remapping."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from defectlens.preprocess.geometry import Affine, Boxes, as_boxes, box_areas, clip_boxes

Image = npt.NDArray[np.uint8]


def crop_roi(
    img: Image,
    boxes: Boxes,
    roi: tuple[int, int, int, int],
    min_visible: float = 0.5,
) -> tuple[Image, Boxes, npt.NDArray[np.bool_], Affine]:
    """Crop ``roi = (x0, y0, x1, y1)`` from ``img``.

    Returns the crop, the boxes shifted into crop coordinates and clipped, a boolean ``keep``
    mask over the *input* boxes, and the :class:`Affine` of the shift. A box is dropped when
    less than ``min_visible`` of its area remains inside the ROI. ``boxes`` in the result
    contain only the kept boxes, in input order.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        msg = f"ROI {roi} invalid for image {w}x{h}"
        raise ValueError(msg)
    shift = Affine(1.0, -float(x0), -float(y0))
    b = as_boxes(boxes)
    shifted = shift.apply(b)
    clipped = clip_boxes(shifted, x1 - x0, y1 - y0)
    full = box_areas(shifted)
    keep = box_areas(clipped) >= min_visible * np.maximum(full, 1e-12)
    keep &= box_areas(clipped) > 0
    return img[y0:y1, x0:x1], clipped[keep], keep, shift
