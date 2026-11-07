"""Three-model anomaly comparison and the few-shot curve, built from MLflow runs.

Every number comes from logged runs (the ``defectlens`` experiment, scored on
``threshold_calibration``). Full-data runs are the latest finished run called exactly
``<model>_<category>`` of phase 3a/3b; few-shot runs carry the tag ``phase=3c``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MODELS = ("patchcore", "anomaly_dino", "efficientad")
LABELS = {"patchcore": "PatchCore", "anomaly_dino": "AnomalyDINO", "efficientad": "EfficientAD"}
# One colour per model, fixed by entity (never by rank): categorical slots 1-3 of the reference
# palette, which validate for all pairs (worst CVD delta-E 9.2, normal vision 24.0).
COLORS = {"anomaly_dino": "#2a78d6", "patchcore": "#eb6834", "efficientad": "#1baf7a"}
METRICS = ("image_auroc", "pixel_auroc", "aupro")
METRIC_LABELS = {"image_auroc": "Image AUROC", "pixel_auroc": "Pixel AUROC", "aupro": "AUPRO"}
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


@dataclass(frozen=True)
class AnomalyRun:
    name: str
    run_id: str
    model: str
    category: str
    phase: str
    shots: int | None
    draw: int | None
    metrics: dict[str, float]
    scalars: dict[str, float] = field(default_factory=dict)


def load_runs(client: Any, experiment_id: str) -> list[AnomalyRun]:
    """Latest finished calibration-scored run per run name."""
    found = client.search_runs(
        [experiment_id],
        filter_string=(
            "tags.eval_split = 'threshold_calibration' and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=5000,
    )
    seen: set[str] = set()
    runs: list[AnomalyRun] = []
    for r in found:
        name = r.info.run_name
        if name in seen:
            continue
        seen.add(name)
        t, m = r.data.tags, r.data.metrics
        runs.append(
            AnomalyRun(
                name=name,
                run_id=r.info.run_id,
                model=t.get("model", ""),
                category=t.get("category", ""),
                phase=t.get("phase", ""),
                shots=int(t["shots"]) if t.get("shots") else None,
                draw=int(t["draw"]) if t.get("draw") else None,
                metrics={
                    k.removeprefix("calib/"): v for k, v in m.items() if k.startswith("calib/")
                },
                scalars={k: v for k, v in m.items() if not k.startswith("calib/")},
            )
        )
    return runs


def full_runs(runs: Sequence[AnomalyRun]) -> dict[tuple[str, str], AnomalyRun]:
    """``{(model, category): run}`` for the full-data runs named exactly ``<model>_<category>``."""
    return {
        (r.model, r.category): r
        for r in runs
        if r.phase in ("3a", "3b") and r.shots is None and r.name == f"{r.model}_{r.category}"
    }


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof 1); the std is NaN for a single value."""
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")


def _fmt(value: float, digits: int = 4) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def full_table(
    full: Mapping[tuple[str, str], AnomalyRun],
    models: Sequence[str],
    categories: Sequence[str],
    metric: str,
) -> str:
    """Rows = categories plus the unweighted mean, columns = models, one metric."""
    head = "| category | " + " | ".join(LABELS[m] for m in models) + " |"
    lines = [head, "|---|" + "---|" * len(models)]
    cols: dict[str, list[float]] = {m: [] for m in models}
    for c in categories:
        cells = []
        for m in models:
            run = full.get((m, c))
            v = run.metrics.get(metric, float("nan")) if run else float("nan")
            cols[m].append(v)
            cells.append(_fmt(v))
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    means = [_fmt(float(np.mean(cols[m]))) for m in models]
    lines.append("| **mean** | " + " | ".join(f"**{x}**" for x in means) + " |")
    return "\n".join(lines)


def cost_table(
    full: Mapping[tuple[str, str], AnomalyRun], models: Sequence[str], categories: Sequence[str]
) -> str:
    """Fit seconds per category and peak VRAM (max over categories) for each model."""
    head = "| model | " + " | ".join(f"fit {c} (s)" for c in categories) + " | peak VRAM (MiB) |"
    lines = [head, "|---|" + "---|" * (len(categories) + 1)]
    for m in models:
        runs = [full.get((m, c)) for c in categories]
        fit = [f"{r.scalars['fit_seconds']:.0f}" if r else "n/a" for r in runs]
        vram = [r.scalars["peak_vram_alloc_mib"] for r in runs if r]
        peak = f"{max(vram):.0f}" if vram else "n/a"
        lines.append(f"| {LABELS[m]} | " + " | ".join(fit) + f" | {peak} |")
    return "\n".join(lines)


def few_shot_values(
    runs: Sequence[AnomalyRun], model: str, metric: str
) -> dict[str, dict[int, dict[int, float]]]:
    """``{category: {shots: {draw: value}}}`` of the phase-3c runs of ``model``."""
    out: dict[str, dict[int, dict[int, float]]] = {}
    for r in runs:
        usable = r.shots is not None and r.draw is not None and metric in r.metrics
        if r.phase == "3c" and r.model == model and usable:
            assert r.shots is not None and r.draw is not None
            out.setdefault(r.category, {}).setdefault(r.shots, {})[r.draw] = r.metrics[metric]
    return out


def few_shot_stats(
    values: Mapping[str, Mapping[int, Mapping[int, float]]],
    categories: Sequence[str],
    shots: Sequence[int],
) -> dict[str, dict[int, tuple[float, float, int]]]:
    """Mean, sample std and number of draws per category and shots, plus a ``"mean"`` entry.

    The ``"mean"`` entry averages the categories within each draw index first (only draw indices
    present in every category count) and then takes mean and std over draws, so its spread is the
    spread of the four-category average, not of the pooled values.
    """
    out: dict[str, dict[int, tuple[float, float, int]]] = {}
    for c in categories:
        out[c] = {}
        for k in shots:
            vals = list(values.get(c, {}).get(k, {}).values())
            if vals:
                m, s = mean_std(vals)
                out[c][k] = (m, s, len(vals))
    out["mean"] = {}
    for k in shots:
        per_cat = [values.get(c, {}).get(k, {}) for c in categories]
        draws = set.intersection(*(set(p) for p in per_cat)) if per_cat else set()
        if draws:
            avg = [float(np.mean([p[d] for p in per_cat])) for d in sorted(draws)]
            m, s = mean_std(avg)
            out["mean"][k] = (m, s, len(draws))
    return out


def few_shot_table(
    stats: Mapping[str, Mapping[int, tuple[float, float, int]]],
    references: Mapping[str, float],
    shots: Sequence[int],
    rows: Sequence[str],
) -> str:
    """Rows = categories (+ ``mean``), columns = shots (``mean ± std``) and the full-data value."""
    head = "| category | " + " | ".join(f"{k}-shot" for k in shots) + " | full data |"
    lines = [head, "|---|" + "---|" * (len(shots) + 1)]
    for c in rows:
        cells = []
        for k in shots:
            st = stats.get(c, {}).get(k)
            cells.append("n/a" if st is None else f"{st[0]:.4f} ± {_fmt(st[1])} (n={st[2]})")
        ref = references.get(c)
        ref_cell = "n/a" if ref is None else _fmt(ref)
        lines.append(f"| {c} | " + " | ".join(cells) + f" | {ref_cell} |")
    return "\n".join(lines)


def plot_few_shot(
    values: Mapping[str, Mapping[int, Mapping[int, float]]],
    stats: Mapping[str, Mapping[int, tuple[float, float, int]]],
    references: Mapping[str, Mapping[str, float]],
    categories: Sequence[str],
    shots: Sequence[int],
    path: Path,
    metric: str = "image_auroc",
) -> None:
    """Few-shot curve (AnomalyDINO) per category and averaged, with full-data reference lines.

    Each panel: faint dots = individual draws, line + caps = mean ± sample std over draws, dashed
    horizontal lines = full-data runs of each model (``references[model][category or "mean"]``).
    The sixth panel holds the legend. Panels have their own y-axes (the categories differ a lot).
    """
    panels = [*categories, "mean"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), facecolor=SURFACE)
    flat = list(axes.ravel())
    dino = COLORS["anomaly_dino"]
    for ax, key in zip(flat, panels, strict=False):
        ax.set_facecolor(SURFACE)
        for m, ref in references.items():
            if key in ref:
                ax.axhline(ref[key], color=COLORS[m], lw=1.6, ls=(0, (5, 3)), zorder=1)
        if key != "mean":
            for k in shots:
                ys = list(values.get(key, {}).get(k, {}).values())
                ax.scatter([k] * len(ys), ys, s=22, color=dino, alpha=0.35, linewidths=0, zorder=2)
        pts = [(k, *stats[key][k][:2]) for k in shots if k in stats.get(key, {})]
        if pts:
            xs, ms, ss = zip(*pts, strict=True)
            err = [0.0 if math.isnan(s) else s for s in ss]
            ax.errorbar(xs, ms, yerr=err, color=dino, lw=2, marker="o", ms=7, capsize=4,
                        mec=SURFACE, mew=1.5, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(list(shots), [str(k) for k in shots])
        ax.minorticks_off()
        ax.set_title("mean of 4 categories" if key == "mean" else key, loc="left",
                     color=INK, fontsize=11)
        ax.set_xlabel("normal training images (shots)", color=INK_2, fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], color=INK_2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    legend_ax = flat[len(panels)]
    legend_ax.axis("off")
    handles = [
        Line2D([], [], color=dino, lw=2, marker="o", ms=7,
               label="AnomalyDINO, k shots\n(mean ± std over draws)"),
        Line2D([], [], color=dino, lw=0, marker="o", ms=5, alpha=0.4, label="one draw"),
    ] + [
        Line2D([], [], color=COLORS[m], lw=1.6, ls=(0, (5, 3)),
               label=f"{LABELS[m]}, all normal images")
        for m in references
    ]
    leg = legend_ax.legend(handles=handles, loc="center left", frameon=False, fontsize=9.5)
    for text in leg.get_texts():
        text.set_color(INK)
    fig.suptitle(f"Few-shot {METRIC_LABELS[metric]} on threshold_calibration", x=0.01, ha="left",
                 color=INK, fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
