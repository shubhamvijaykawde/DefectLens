"""Build the detector comparison from MLflow runs: markdown tables and confusion-matrix figures.

Everything shown comes from logged runs. Validation numbers are read at the run's ``best_epoch``
(the checkpoint that was scored on test), not from the last epoch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from defectlens.eval.confusion import normalize_rows

HEADLINE = ("map50", "map50_95", "recall", "precision", "fp_per_image", "recall_max")


@dataclass
class RunSummary:
    name: str
    run_id: str
    val: dict[str, float]
    test: dict[str, float]
    scalars: dict[str, float]  # best_epoch, epochs_run, train_seconds, peak_vram_alloc_mib
    config: dict[str, Any]
    confusion_md: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


def parse_confusion_md(text: str) -> tuple[list[str], npt.NDArray[np.int64]]:
    """Inverse of ``report.confusion_table``: returns (class names without background, counts)."""
    rows = [ln.strip().strip("|").split("|") for ln in text.strip().splitlines()]
    header = [c.strip() for c in rows[0]][1:]
    body = [[c.strip() for c in r] for r in rows[2:]]
    cm = np.array([[int(v) for v in r[1:]] for r in body], dtype=np.int64)
    if header[-1] != "background" or cm.shape != (len(header), len(header)):
        msg = "not a confusion table with a background row/column"
        raise ValueError(msg)
    return header[:-1], cm


def load_run(client: Any, experiment_id: str, name: str) -> RunSummary:
    """Latest FINISHED run called ``name``. ``val`` is taken at ``best_epoch`` when logged."""
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.runName = '{name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        msg = f"no finished run named {name!r}"
        raise LookupError(msg)
    run = runs[0]
    metrics = run.data.metrics
    best = metrics.get("best_epoch")
    val: dict[str, float] = {}
    test: dict[str, float] = {}
    for key, last in metrics.items():
        split, _, metric = key.partition("/")
        if split == "test":
            test[metric] = last
        elif split == "val":
            if best is None:
                val[metric] = last
            else:
                hist = {m.step: m.value for m in client.get_metric_history(run.info.run_id, key)}
                if int(best) in hist:
                    val[metric] = hist[int(best)]
    arts = {a.path for a in client.list_artifacts(run.info.run_id)}
    cm_md = None
    if "test_confusion.md" in arts:
        local = client.download_artifacts(run.info.run_id, "test_confusion.md")
        cm_md = Path(local).read_text()
    cfg_path = client.download_artifacts(run.info.run_id, "config.json")
    import json

    scalars = {k: float(v) for k, v in metrics.items()
               if "/" not in k and k in ("best_epoch", "epochs_run", "train_seconds",
                                          "peak_vram_alloc_mib")}
    return RunSummary(name, run.info.run_id, val, test, scalars,
                      json.loads(Path(cfg_path).read_text()), cm_md, dict(run.data.tags))


def _f(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.{nd}f}"


def headline_table(runs: Sequence[RunSummary]) -> str:
    cols = ("AP@50", "AP@50:95", "recall@0.5", "precision@0.5", "FP/img@0.5", "recall_max")
    lines = ["| detector | split | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in runs:
        for split, m in (("val", r.val), ("test", r.test)):
            lines.append(f"| {r.name} | {split} | "
                         + " | ".join(_f(m.get(k)) for k in HEADLINE) + " |")
    return "\n".join(lines)


def per_class_table(runs: Sequence[RunSummary], class_names: Sequence[str]) -> str:
    """Test AP@50 and recall@0.5 per class, one column pair per run (class-aware runs only)."""
    deep = [r for r in runs if f"ap50_{class_names[0]}" in r.test]
    head = "| class | " + " | ".join(f"{r.name} AP@50 | {r.name} recall" for r in deep) + " |"
    lines = [head, "|---|" + "---|---|" * len(deep)]
    for c in class_names:
        cells = [f"{_f(r.test.get(f'ap50_{c}'))} | {_f(r.test.get(f'recall_{c}'))}" for r in deep]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def protocol_table(runs: Sequence[RunSummary]) -> str:
    """What each run was trained with, read from its logged config (not typed by hand)."""
    lines = ["| detector | input | batch (x accum) | epochs max / run | early stopping | "
             "augmentation | best epoch | train time | peak VRAM |", "|---|" + "---|" * 8]
    for r in runs:
        m = r.config.get("model") or {}
        if "augment" not in m:
            lines.append(f"| {r.name} | n/a | n/a | n/a | n/a | n/a (non-learned) | n/a | "
                         f"{_f(r.scalars.get('train_seconds'), 0)} s | n/a |")
            continue
        aug = ", ".join(f"{k} {v}" for k, v in m["augment"].items())
        pat = m.get("patience")
        run_ep = r.scalars.get("epochs_run")
        best = r.scalars.get("best_epoch")
        lines.append(
            f"| {r.name} | {m['input_size']} | {m['batch_size']} x {m.get('accum_steps', 1)} | "
            f"{m['epochs']} / {int(run_ep) if run_ep is not None else 'not logged'} | "
            f"{'patience ' + str(pat) if pat is not None else 'off'} | {aug} | "
            f"{int(best) + 1 if best is not None else 'n/a'} | "
            f"{_f(r.scalars.get('train_seconds'), 0)} s | "
            f"{_f(r.scalars.get('peak_vram_alloc_mib'), 0)} MiB |")
    return "\n".join(lines)


def plot_confusion(
    cm: npt.NDArray[np.int64], class_names: Sequence[str], title: str, path: Path
) -> None:
    """Row-normalised heat map (one sequential blue ramp) with raw counts printed in the cells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = [*class_names, "background"]
    norm = normalize_rows(cm)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f1f6fb", "#9ec3e6", "#2a6db0", "#0b2f5b"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    ax.imshow(norm, cmap=cmap, vmin=0.0, vmax=1.0)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.5 else "#1a1a1a")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted (score >= 0.5)")
    ax.set_ylabel("true (last row: false alarms on background)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.axhline(len(names) - 1.5, color="#888888", lw=0.8)
    ax.axvline(len(names) - 1.5, color="#888888", lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.01, 0.005, "cell colour = share of the true-class row; numbers = counts",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def confusion_summary(cm: npt.NDArray[np.int64], class_names: Sequence[str]) -> Mapping[str, int]:
    """Totals used in the text: matched with correct class, wrong class, missed, false alarms."""
    k = len(class_names)
    diag = int(np.trace(cm[:k, :k]))
    return {"correct": diag, "wrong_class": int(cm[:k, :k].sum()) - diag,
            "missed": int(cm[:k, k].sum()), "false_alarms": int(cm[k, :k].sum())}
