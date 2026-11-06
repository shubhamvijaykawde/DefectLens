"""Turn metric dicts and confusion matrices into markdown, and log them to MLflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

HEADLINE = ("map50", "map50_95", "recall", "precision", "fp_per_image", "recall_max")


def metrics_table(
    by_split: Mapping[str, Mapping[str, float]], keys: Sequence[str] = HEADLINE
) -> str:
    """Markdown table: one row per metric, one column per split."""
    splits = list(by_split)
    lines = ["| metric | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
    for k in keys:
        cells = [f"{by_split[s][k]:.4f}" if k in by_split[s] else "n/a" for s in splits]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_class_table(metrics: Mapping[str, float], class_names: Sequence[str]) -> str:
    """Markdown table of AP@50 and recall per class."""
    lines = ["| class | AP@50 | recall |", "|---|---|---|"]
    for n in class_names:
        lines.append(f"| {n} | {metrics[f'ap50_{n}']:.4f} | {metrics[f'recall_{n}']:.4f} |")
    return "\n".join(lines)


def confusion_table(cm: npt.NDArray[np.int64], class_names: Sequence[str]) -> str:
    """Markdown table; rows are true class (last row: background), columns are predictions."""
    names = [*class_names, "background"]
    lines = ["| true \\ pred | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for i, n in enumerate(names):
        lines.append(f"| {n} | " + " | ".join(str(int(v)) for v in cm[i]) + " |")
    return "\n".join(lines)


def log_metrics(prefix: str, metrics: Mapping[str, float], step: int | None = None) -> None:
    """Log to the active MLflow run as ``<prefix>/<metric>``; NaNs are skipped."""
    import mlflow

    clean = {f"{prefix}/{k}": float(v) for k, v in metrics.items() if not np.isnan(v)}
    mlflow.log_metrics(clean, step=step)
