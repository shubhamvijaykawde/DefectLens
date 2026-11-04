"""Box-aware geometry shared by every preprocessing step.

Boxes are float arrays of shape (N, 4) in ``xyxy`` absolute pixels. Every geometric step is a
uniform scale plus translation, so the whole chain composes into one :class:`Affine` that also
maps predictions back to original-image coordinates when serving.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Boxes = npt.NDArray[np.float64]


def as_boxes(boxes: npt.ArrayLike | None) -> Boxes:
    """Coerce to a float (N, 4) array; ``None`` becomes an empty (0, 4) array."""
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float64)
    arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return arr


@dataclass(frozen=True)
class Affine:
    """``x' = s * x + tx``, ``y' = s * y + ty`` (uniform scale, so boxes stay boxes)."""

    s: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def then(self, other: Affine) -> Affine:
        """Apply ``self`` first, then ``other``."""
        return Affine(
            other.s * self.s,
            other.s * self.tx + other.tx,
            other.s * self.ty + other.ty,
        )

    def apply(self, boxes: Boxes) -> Boxes:
        out = as_boxes(boxes) * self.s
        out[:, [0, 2]] += self.tx
        out[:, [1, 3]] += self.ty
        return out

    def invert(self, boxes: Boxes) -> Boxes:
        out = as_boxes(boxes).copy()
        out[:, [0, 2]] -= self.tx
        out[:, [1, 3]] -= self.ty
        return np.asarray(out / self.s, dtype=np.float64)


def clip_boxes(boxes: Boxes, width: float, height: float) -> Boxes:
    """Clip to ``[0, width] x [0, height]``."""
    out = as_boxes(boxes).copy()
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out


def box_areas(boxes: Boxes) -> npt.NDArray[np.float64]:
    b = as_boxes(boxes)
    w = np.clip(b[:, 2] - b[:, 0], 0, None)
    h = np.clip(b[:, 3] - b[:, 1], 0, None)
    return np.asarray(w * h, dtype=np.float64)
