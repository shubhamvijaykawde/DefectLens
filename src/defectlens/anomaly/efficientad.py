"""EfficientAD (small) trained with our own loop on top of anomalib's ``EfficientAdModel``.

anomalib's ``EfficientAd`` Lightning module is not used: its splits, validation quantiles and
Engine are not ours. We reuse only the torch network and its loss/map code
(teacher, student, autoencoder, hard-example loss, ImageNet penalty) and re-implement the short
training procedure of the method:

* the teacher is the pretrained patch-description network shipped with anomalib (downloaded once,
  SHA-256 checked by anomalib), frozen; EfficientAD has its own network, no backbone choice;
* batch size 1 on the normal images, plus one random Imagenette image per step for the penalty
  loss; Adam, lr 1e-4, weight decay 1e-5, lr x0.1 at 95% of the steps;
* teacher channel mean/std come from the training images; the map-normalisation quantiles
  (90% and 99.5%) come from a held-out 10% of the *training* normals, so no calibration or
  evaluation image is used for fitting or normalisation (anomalib uses its validation split).

The image score is the maximum of the combined (student-teacher + student-autoencoder) map.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from anomalib.data import InferenceBatch
from anomalib.data.utils import download_and_extract
from anomalib.models.image.efficient_ad.lightning_model import (
    IMAGENETTE_DOWNLOAD_INFO,
    WEIGHTS_DOWNLOAD_INFO,
)
from anomalib.models.image.efficient_ad.torch_model import (
    EfficientAdModel,
    EfficientAdModelSize,
    reduce_tensor_elems,
)
from anomalib.utils.path import get_pretrained_weights_dir
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from defectlens.anomaly.datamodule import IMAGENET_MEAN, IMAGENET_STD

MIN_IMAGE_SIZE = 256  # smaller inputs fail inside the autoencoder (checked: 224 does not run)


def teacher_weights_path(model_size: str) -> Path:
    """Path of the pretrained teacher weights, downloading anomalib's release zip if needed."""
    weights_dir = get_pretrained_weights_dir()
    folder = weights_dir / "efficientad_pretrained_weights"
    if not folder.is_dir():
        download_and_extract(weights_dir, WEIGHTS_DOWNLOAD_INFO)
    return folder / f"pretrained_teacher_{model_size}.pth"


def ensure_imagenette(root: Path) -> None:
    """Download and extract Imagenette (about 1.5 GB) into ``root`` unless ``root`` exists.

    The download goes to ``<root>.partial`` and is renamed on success, so an interrupted download
    never leaves a half-filled ``root`` behind (a stale ``.partial`` is deleted and restarted).
    """
    if root.is_dir():
        return
    partial = root.with_name(root.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    download_and_extract(partial, IMAGENETTE_DOWNLOAD_INFO)  # hash is checked after the download
    partial.rename(root)


def penalty_loader(root: Path, image_size: tuple[int, int], num_workers: int) -> DataLoader[object]:
    """Batch-size-1 shuffled loader of Imagenette images in [0, 1] for the penalty loss.

    Same transforms as anomalib: resize to twice the size, random grayscale (p 0.3), centre crop.
    """
    h, w = image_size
    transform = v2.Compose(
        [
            v2.Resize((2 * h, 2 * w)),
            v2.RandomGrayscale(p=0.3),
            v2.CenterCrop((h, w)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    return DataLoader(
        ImageFolder(str(root), transform=transform),
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
    )


def endless(batches: Iterable[object]) -> Iterator[torch.Tensor]:
    """Images of an (image, label) loader or a plain tensor iterable, restarting when exhausted."""
    while True:
        for item in batches:
            tensor = item[0] if isinstance(item, list | tuple) else item
            assert isinstance(tensor, torch.Tensor)
            yield tensor


class EfficientAD:
    """Fit on normal-image batches (any batch size), then score batches.

    ``fit`` takes the same ImageNet-normalised batches as the memory-bank models; they are
    converted back to [0, 1] on the device because the network normalises internally.
    """

    def __init__(
        self,
        model_size: str = "small",
        teacher_out_channels: int = 384,
        pretrained_teacher: bool = True,
        imagenet_dir: Path | None = None,
        image_size: tuple[int, int] = (256, 256),
        max_steps: int = 70_000,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        train_batch_size: int = 1,
        holdout_fraction: float = 0.1,
        log_every: int = 100,
        num_workers: int = 2,
        device: str = "cuda",
        seed: int = 42,
        penalty_images: Iterable[object] | None = None,
    ) -> None:
        if train_batch_size != 1:
            msg = f"EfficientAD trains with batch size 1, got train_batch_size={train_batch_size}"
            raise ValueError(msg)
        if min(image_size) < MIN_IMAGE_SIZE:
            msg = (
                f"image_size {tuple(image_size)} is too small: the autoencoder downsamples by 32 "
                f"and then applies an 8x8 kernel, so each side must be >= {MIN_IMAGE_SIZE}"
            )
            raise ValueError(msg)
        if max_steps < 1:
            msg = f"max_steps must be at least 1, got {max_steps}"
            raise ValueError(msg)
        if not 0.0 < holdout_fraction < 1.0:
            msg = f"holdout_fraction must be in (0, 1), got {holdout_fraction}"
            raise ValueError(msg)
        if penalty_images is None and imagenet_dir is None:
            msg = "need imagenet_dir or penalty_images for the penalty loss"
            raise ValueError(msg)
        self.device = device
        self.seed = seed
        self.imagenet_dir = imagenet_dir
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.max_steps = max_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.holdout_fraction = holdout_fraction
        self.log_every = max(1, log_every)
        self.num_workers = num_workers
        self._penalty_images = penalty_images
        size = EfficientAdModelSize(model_size)
        self.model = EfficientAdModel(
            teacher_out_channels=teacher_out_channels, model_size=size
        ).to(device)
        if pretrained_teacher:
            path = teacher_weights_path(size.value)
            self.model.teacher.load_state_dict(
                torch.load(path, map_location=device, weights_only=True)
            )
        self.model.eval()
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.history: list[dict[str, float]] = []
        self._info: dict[str, float] = {}

    def _to_unit(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalised batch back to [0, 1] (the network applies its own normalisation)."""
        return x.to(self.device) * self._std + self._mean

    def _penalty_iterator(self) -> Iterator[torch.Tensor]:
        if self._penalty_images is not None:
            return endless(self._penalty_images)
        assert self.imagenet_dir is not None
        ensure_imagenette(self.imagenet_dir)
        return endless(penalty_loader(self.imagenet_dir, self.image_size, self.num_workers))

    @torch.no_grad()
    def _teacher_mean_std(self, images: torch.Tensor, index: torch.Tensor) -> None:
        """Channel-wise mean and std of the teacher output over the training images."""
        count = 0
        total: torch.Tensor | None = None
        total_sq: torch.Tensor | None = None
        for chunk in torch.split(index, 16):
            y = self.model.teacher(self._to_unit(images[chunk])).double()
            if total is None or total_sq is None:
                total = torch.zeros(y.shape[1], dtype=torch.float64, device=y.device)
                total_sq = torch.zeros_like(total)
            count += y.shape[0] * y.shape[2] * y.shape[3]
            total += y.sum(dim=(0, 2, 3))
            total_sq += (y**2).sum(dim=(0, 2, 3))
        assert total is not None and total_sq is not None
        mean = total / count
        std = (total_sq / count - mean**2).clamp_min(0).sqrt()
        self.model.mean_std["mean"].copy_(mean.float().view(1, -1, 1, 1))
        self.model.mean_std["std"].copy_(std.float().view(1, -1, 1, 1))

    @torch.no_grad()
    def _map_quantiles(self, images: torch.Tensor, index: torch.Tensor) -> None:
        """90% / 99.5% quantiles of the raw student and autoencoder maps on held-out normals."""
        maps_st, maps_ae = [], []
        for i in index.tolist():
            st, ae = self.model.get_maps(self._to_unit(images[i : i + 1]), normalize=False)
            maps_st.append(st)
            maps_ae.append(ae)
        for key, maps in (("st", maps_st), ("ae", maps_ae)):
            flat = reduce_tensor_elems(torch.cat(maps))
            self.model.quantiles[f"qa_{key}"].copy_(torch.quantile(flat, 0.9))
            self.model.quantiles[f"qb_{key}"].copy_(torch.quantile(flat, 0.995))

    def fit(self, batches: Iterable[torch.Tensor]) -> None:
        # Start the penalty-image workers before the training images are materialised.
        penalty = self._penalty_iterator()
        chunks = [b.cpu() for b in batches]
        if not chunks:
            msg = "no training batches"
            raise ValueError(msg)
        images = torch.cat(chunks)
        del chunks
        n = images.shape[0]
        if n < 2:
            msg = f"need at least 2 training images (one is held out for normalisation), got {n}"
            raise ValueError(msg)

        gen = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=gen)
        n_hold = max(1, round(self.holdout_fraction * n))
        holdout, train = perm[:n_hold], perm[n_hold:]

        net = self.model
        self._teacher_mean_std(images, train)

        net.train()
        net.teacher.eval()
        params = list(net.student.parameters()) + list(net.ae.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(1, int(0.95 * self.max_steps)), gamma=0.1
        )

        self.history = []
        window = torch.zeros(3, device=self.device)
        order: list[int] = []
        t0 = time.perf_counter()
        for step in range(1, self.max_steps + 1):
            if not order:
                order = train[torch.randperm(len(train), generator=gen)].tolist()
            x = self._to_unit(images[[order.pop()]])  # batch size 1
            batch_imagenet = next(penalty).to(self.device)
            out = net(x, batch_imagenet)
            if isinstance(out, InferenceBatch):
                msg = "model returned inference output in training mode"
                raise RuntimeError(msg)
            loss_st, loss_ae, loss_stae = out
            loss = loss_st + loss_ae + loss_stae
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            window += torch.stack([loss_st, loss_ae, loss_stae]).detach()
            if step % self.log_every == 0 or step == self.max_steps:
                span = self.log_every if step % self.log_every == 0 else step % self.log_every
                mean = (window / span).tolist()
                self.history.append(
                    {"step": float(step), "loss_st": mean[0], "loss_ae": mean[1],
                     "loss_stae": mean[2], "lr": float(sched.get_last_lr()[0])}
                )
                window.zero_()
                if step % (self.log_every * 10) == 0 or step == self.max_steps:
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / step * (self.max_steps - step)
                    print(
                        f"  step {step}/{self.max_steps} loss_st={mean[0]:.4f} "
                        f"loss_ae={mean[1]:.4f} loss_stae={mean[2]:.4f} "
                        f"{step / elapsed:.1f} it/s ETA {eta / 60:.0f} min",
                        flush=True,
                    )
        train_s = time.perf_counter() - t0

        net.eval()
        self._map_quantiles(images, holdout)
        last = self.history[-1]
        self._info = {
            "n_trained": float(len(train)),
            "n_holdout": float(n_hold),
            "train_steps": float(self.max_steps),
            "train_seconds": train_s,
            "steps_per_second": self.max_steps / train_s,
            "final_loss_st": last["loss_st"],
            "final_loss_ae": last["loss_ae"],
            "final_loss_stae": last["loss_stae"],
        }

    def fit_info(self) -> dict[str, float]:
        """Training summary after ``fit`` (image counts, steps, speed, last loss window)."""
        return dict(self._info)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Teacher, student, autoencoder, teacher mean/std and map quantiles (CPU tensors)."""
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        out = self.model(self._to_unit(images))
        if not isinstance(out, InferenceBatch):
            msg = "model returned training output in eval mode"
            raise RuntimeError(msg)
        assert out.pred_score is not None and out.anomaly_map is not None
        return out.pred_score.reshape(-1), out.anomaly_map.squeeze(1)
