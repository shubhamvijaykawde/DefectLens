"""AnomalyDINO (DINOv2-small) on top of anomalib's ``AnomalyDINOModel``.

The patch memory bank is the L2-normalised last-block DINOv2 patch tokens of the normal images
(no coreset by default); scoring is nearest-neighbour cosine distance. Few-shot use is just a
smaller training set (see ``few_shot_subset``).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel


class AnomalyDino:
    def __init__(
        self,
        encoder_name: str = "vit_small_patch14_dinov2",
        num_neighbours: int = 1,
        masking: bool = False,
        coreset_subsampling: bool = False,
        sampling_ratio: float = 0.1,
        device: str = "cuda",
        model: AnomalyDINOModel | None = None,
    ) -> None:
        self.device = device
        self.model = (
            model
            if model is not None
            else AnomalyDINOModel(
                num_neighbours=num_neighbours,
                encoder_name=encoder_name,
                masking=masking,
                coreset_subsampling=coreset_subsampling,
                sampling_ratio=sampling_ratio,
            )
        ).to(device).eval()

    @torch.no_grad()
    def fit(self, batches: Iterable[torch.Tensor]) -> None:
        """Collect patch embeddings in training mode, then finalise the memory bank."""
        self.model.embedding_store.clear()
        self.model.train()  # the model only stores embeddings in train mode; the encoder is frozen
        try:
            for x in batches:
                self.model(x.to(self.device))
            self.model.fit()
        finally:
            self.model.eval()

    @property
    def memory_bank(self) -> torch.Tensor:
        """The fitted patch memory bank, (rows, dim)."""
        bank: torch.Tensor = self.model.memory_bank
        return bank

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(images.to(self.device))
        return out.pred_score.reshape(-1), out.anomaly_map.squeeze(1)
