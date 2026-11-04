"""VisA ``1cls`` splits: train (normal only) plus two disjoint halves of the official test set.

``threshold_calibration`` may be used to choose thresholds and models; ``threshold_evaluation``
is for one final frozen-threshold evaluation and nothing else (see PLAN.md, Section 3).
"""

from __future__ import annotations

import csv
from pathlib import Path

from defectlens.data.splits import stratified_split

CATEGORIES: tuple[str, ...] = ("cashew", "candle", "pcb1", "macaroni2")


def read_one_cls(csv_path: Path, category: str) -> list[dict[str, str]]:
    """Rows of the official ``split_csv/1cls.csv`` for one category."""
    with csv_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["object"] == category]
    if not rows:
        msg = f"no rows for category {category!r} in {csv_path}"
        raise ValueError(msg)
    return rows


def make_visa_splits(
    rows: list[dict[str, str]], seed: int
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return ``({train, threshold_calibration, threshold_evaluation}, labels)`` where values
    are image paths relative to the VisA root and ``labels`` maps path -> normal|anomaly.

    The official test rows are split 50/50, stratified by label. With an odd count the
    calibration half gets the extra image (the first part takes ``round(n / 2)``).
    """
    labels = {r["image"]: r["label"] for r in rows}
    train = sorted(r["image"] for r in rows if r["split"] == "train")
    if any(labels[p] != "normal" for p in train):
        msg = "train split contains non-normal images"
        raise ValueError(msg)
    test = [(r["image"], r["label"]) for r in rows if r["split"] == "test"]
    halves = stratified_split(
        test, {"threshold_calibration": 0.5, "threshold_evaluation": 0.5}, seed
    )
    return {"train": train, **halves}, labels
