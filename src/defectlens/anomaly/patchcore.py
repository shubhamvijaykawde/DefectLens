"""PatchCore (wide_resnet50_2 baseline) on top of anomalib's ``PatchcoreModel``.

Only the memory-bank construction is our own: anomalib's version keeps every training patch
embedding on the GPU and stacks them, which is about 4.3 GB (plus a copy) for 900 images at
224 px (arithmetic, 900 x 784 x 1536 x 4 bytes). Here embeddings are kept in CPU RAM, projected to a
low dimension on the GPU, and the greedy k-center coreset runs on the projections (the same
approximation PatchCore uses). Scoring is anomalib's own nearest-neighbour code.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def greedy_coreset(z: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Indices of ``n`` points chosen by greedy k-center selection on ``z`` of shape (N, d).

    Starts from a seeded random point, then repeatedly adds the point farthest from the chosen set.
    Returns sorted unique indices. ``n >= N`` returns every index.
    """
    total = z.shape[0]
    if n >= total:
        return torch.arange(total)
    gen = torch.Generator().manual_seed(seed)
    current = int(torch.randint(total, (1,), generator=gen))
    chosen = torch.empty(n, dtype=torch.long)
    min_dist = torch.full((total,), float("inf"), device=z.device)
    for i in range(n):
        chosen[i] = current
        dist = (z - z[current]).pow(2).sum(dim=1)
        min_dist = torch.minimum(min_dist, dist)
        current = int(torch.argmax(min_dist))
    unique: torch.Tensor = torch.unique(chosen)  # sorted
    return unique


def gather_rows(chunks: Sequence[torch.Tensor], index: torch.Tensor) -> torch.Tensor:
    """Rows ``index`` (sorted) of the virtual concatenation of ``chunks`` without concatenating."""
    out, start = [], 0
    for chunk in chunks:
        end = start + chunk.shape[0]
        local = index[(index >= start) & (index < end)] - start
        if local.numel():
            out.append(chunk[local])
        start = end
    return torch.cat(out)


class PatchCore:
    """Fit a coreset memory bank on normal images; score with k-nearest-neighbour distances."""

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        coreset_ratio: float = 0.01,
        num_neighbors: int = 9,
        projection_dim: int = 128,
        pre_trained: bool = True,
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        self.device = device
        self.seed = seed
        self.coreset_ratio = coreset_ratio
        self.projection_dim = projection_dim
        self.model = PatchcoreModel(
            layers=list(layers), backbone=backbone, pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        ).to(device).eval()

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Patch embeddings of shape (B * h * w, D) for a normalised image batch."""
        m = self.model
        features = m.feature_extractor(images.to(self.device))
        features = {k: m.feature_pooler(v) for k, v in features.items()}
        return m.reshape_embedding(m.generate_embedding(features))

    @torch.no_grad()
    def fit(self, batches: Iterable[torch.Tensor]) -> None:
        chunks: list[torch.Tensor] = []
        projected: list[torch.Tensor] = []
        proj: torch.Tensor | None = None
        for x in batches:
            emb = self.embed(x)
            if proj is None:
                gen = torch.Generator().manual_seed(self.seed)
                proj = (
                    torch.randn(emb.shape[1], self.projection_dim, generator=gen)
                    / self.projection_dim**0.5
                ).to(self.device)
            projected.append(emb @ proj)
            chunks.append(emb.cpu())
        if not chunks:
            msg = "no training batches"
            raise ValueError(msg)
        z = torch.cat(projected)
        n = max(1, round(self.coreset_ratio * z.shape[0]))
        index = greedy_coreset(z, n, self.seed).to("cpu")
        self.model.memory_bank = gather_rows(chunks, index).to(self.device)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(images.to(self.device))
        return out.pred_score.reshape(-1), out.anomaly_map.squeeze(1)
