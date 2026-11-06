"""Threshold-free anomaly metrics: image AUROC, pixel AUROC and AUPRO, plus a heatmap grid.

All metrics take plain numpy arrays so they run on CPU and are testable with tiny fixtures.
``scores`` are higher = more anomalous; labels and masks are 1 = anomalous.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy.stats import rankdata

FloatArr = npt.NDArray[np.floating]
IntArr = npt.NDArray[np.integer]
BoolArr = npt.NDArray[np.bool_]


def auroc(scores: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) statistic, ties count 1/2.

    Returns NaN when only one class is present.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)  # average ranks for ties
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def aupro(
    maps: FloatArr, masks: BoolArr, fpr_limit: float = 0.3, n_bins: int = 4096
) -> float:
    """Area under the per-region-overlap curve for FPR in [0, ``fpr_limit``], normalised to [0, 1].

    For each threshold, FPR is measured on all pixels outside the ground-truth masks and PRO is
    the mean over connected ground-truth regions of the fraction of that region scored above the
    threshold. Thresholds are ``n_bins`` equal-width levels between the min and max score, so the
    result is approximate when scores need finer resolution than that.

    ``maps`` and ``masks`` have shape (N, H, W). Returns NaN with no anomalous region or no
    normal pixel.
    """
    maps = np.asarray(maps, dtype=np.float64)
    masks = np.asarray(masks).astype(bool)
    if maps.shape != masks.shape:
        msg = f"maps {maps.shape} and masks {masks.shape} differ"
        raise ValueError(msg)
    lo, hi = float(maps.min()), float(maps.max())
    if hi <= lo or not masks.any() or masks.all():
        return float("nan")
    edges = np.linspace(lo, hi, n_bins + 1)

    def exceed_counts(values: FloatArr) -> FloatArr:
        """Number of values >= edges[i], for i in 0..n_bins (decreasing in i)."""
        idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins)
        hist = np.bincount(idx, minlength=n_bins + 1)
        return np.cumsum(hist[::-1])[::-1].astype(np.float64)

    # Threshold i keeps pixels with score >= edges[i]; i = 0 keeps everything, i = n_bins the max.
    normal_pixels = maps[~masks]
    fpr = exceed_counts(normal_pixels) / normal_pixels.size
    pro_sum = np.zeros(n_bins + 1)
    n_regions = 0
    for mask, amap in zip(masks, maps, strict=True):
        labelled, n = ndimage.label(mask)
        for r in range(1, n + 1):
            region = amap[labelled == r]
            pro_sum += exceed_counts(region) / region.size
            n_regions += 1
    pro = pro_sum / n_regions

    # Order by increasing FPR (threshold decreasing in score = increasing index reversed).
    order = np.argsort(fpr, kind="stable")
    fpr, pro = fpr[order], pro[order]
    keep = fpr <= fpr_limit
    f, p = fpr[keep], pro[keep]
    if f[-1] < fpr_limit and keep.sum() < fpr.size:  # interpolate to the limit
        nxt = np.flatnonzero(~keep)[0]
        p_lim = np.interp(fpr_limit, [fpr[nxt - 1], fpr[nxt]], [pro[nxt - 1], pro[nxt]])
        f, p = np.append(f, fpr_limit), np.append(p, p_lim)
    if f[0] > 0:  # the curve starts at FPR 0 with the same PRO as its first point
        f, p = np.insert(f, 0, 0.0), np.insert(p, 0, p[0])
    return float(np.sum(np.diff(f) * (p[1:] + p[:-1]) / 2) / fpr_limit)


def evaluate_anomaly(
    image_scores: FloatArr, labels: IntArr, maps: FloatArr, masks: BoolArr
) -> dict[str, float]:
    """Image AUROC, pixel AUROC (all pixels pooled) and AUPRO for one split.

    Pixel metrics use every image; normal images have an all-zero mask.
    """
    return {
        "image_auroc": auroc(image_scores, labels),
        "pixel_auroc": auroc(maps.reshape(-1), masks.reshape(-1)),
        "aupro": aupro(maps, masks),
    }


def save_heatmap_grid(
    images: FloatArr,
    maps: FloatArr,
    masks: BoolArr,
    scores: FloatArr,
    labels: IntArr,
    path: Path,
    title: str,
    n_per_kind: int = 4,
) -> None:
    """Save a grid: image | heatmap overlay | ground truth, for the most- and least-anomalous
    images of each label (so both confident hits and misses show).

    ``images`` are (N, H, W, 3) floats in [0, 1].
    """
    pick: list[int] = []
    for lab in (1, 0):
        idx = np.flatnonzero(labels == lab)
        order = idx[np.argsort(scores[idx])]
        # anomalous: highest scores first (hits) ; normal: highest scores first (false alarms)
        pick += list(order[::-1][:n_per_kind])
    lo, hi = float(maps.min()), float(maps.max())
    fig, axes = plt.subplots(len(pick), 3, figsize=(7.5, 2.4 * len(pick)), squeeze=False)
    for row, i in enumerate(pick):
        axes[row, 0].imshow(images[i])
        axes[row, 1].imshow(images[i])
        axes[row, 1].imshow(maps[i], cmap="jet", alpha=0.5, vmin=lo, vmax=hi)
        axes[row, 2].imshow(masks[i], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"{'anomaly' if labels[i] else 'normal'}\nscore {scores[i]:.3f}")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, name in zip(axes[0], ("image", "anomaly map", "ground truth"), strict=True):
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
