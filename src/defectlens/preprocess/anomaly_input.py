"""Input preparation of the anomaly models, shared by training / evaluation and the API.

The anomaly models take a square-resized RGB image, scaled to [0, 1] and normalised with ImageNet
statistics (``VisaImages`` in ``anomaly/datamodule.py`` and ``serve/inference.py`` both call
``resize_normalize``). numpy and OpenCV only, so the slim serving image can import it without torch.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_normalize(
    bgr: npt.NDArray[Any], image_size: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """BGR uint8 ``(H0, W0, 3)`` -> RGB float32 ``(H, W, 3)``, ImageNet-normalised.

    ``image_size`` is ``(height, width)``. The aspect ratio is not kept (the models were fitted on
    stretched images); INTER_AREA is the interpolation of the training and evaluation path.
    """
    h, w = image_size
    rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return np.asarray((rgb.astype(np.float32) / 255.0 - mean) / std, dtype=np.float32)
