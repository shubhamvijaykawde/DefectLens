"""Common interface of the memory-bank anomaly models and the prediction loop."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch

from defectlens.anomaly.datamodule import denormalize


class AnomalyModel(Protocol):
    """Fit on normal-image batches, then score batches."""

    def fit(self, batches: Iterable[torch.Tensor]) -> None:
        """Build the memory bank from (B,3,H,W) batches of normal images."""
        ...

    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image scores (B,) and anomaly maps (B,H,W) at the input resolution."""
        ...


@dataclass
class Predictions:
    """Everything needed to score and plot one split (all on CPU, numpy)."""

    scores: npt.NDArray[np.float32]  # (N,)
    maps: npt.NDArray[np.float32]  # (N,H,W)
    masks: npt.NDArray[np.bool_]  # (N,H,W)
    labels: npt.NDArray[np.int64]  # (N,)
    images: npt.NDArray[np.float32]  # (N,H,W,3) in [0,1]
    paths: list[str]


@torch.no_grad()
def collect_predictions(
    model: AnomalyModel, loader: Iterable[dict[str, Any]], device: str
) -> Predictions:
    """Run ``model`` over a loader of ``VisaImages`` batches."""
    scores, maps, masks, labels, images, paths = [], [], [], [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        s, m = model.predict(x)
        scores.append(s.float().cpu().numpy())
        maps.append(m.float().cpu().numpy())
        masks.append(batch["mask"].numpy() > 0.5)
        labels.append(np.asarray(batch["label"]))
        images.append(denormalize(batch["image"]))
        paths += list(batch["path"])
    return Predictions(
        scores=np.concatenate(scores).astype(np.float32),
        maps=np.concatenate(maps).astype(np.float32),
        masks=np.concatenate(masks),
        labels=np.concatenate(labels).astype(np.int64),
        images=np.concatenate(images),
        paths=paths,
    )
