from pathlib import Path

import numpy as np
import pytest

from defectlens.eval.compare import (
    RunSummary,
    confusion_summary,
    headline_table,
    parse_confusion_md,
    per_class_table,
    plot_confusion,
    protocol_table,
)
from defectlens.eval.report import confusion_table

NAMES = ["a", "b"]
CM = np.array([[5, 1, 2], [0, 4, 3], [6, 7, 0]], dtype=np.int64)


def _run(name: str, **test: float) -> RunSummary:
    cfg = {"model": {"augment": {"hflip": 0.5}, "input_size": 64, "batch_size": 2, "epochs": 5,
                     "patience": 2, "accum_steps": 2}}
    return RunSummary(name, "id", {"map50": 0.5}, {"map50": 0.6, **test},
                      {"best_epoch": 3.0, "epochs_run": 4.0, "train_seconds": 10.0}, cfg)


def test_confusion_table_round_trips() -> None:
    names, cm = parse_confusion_md(confusion_table(CM, NAMES))
    assert names == NAMES
    np.testing.assert_array_equal(cm, CM)


def test_parse_rejects_non_confusion_table() -> None:
    with pytest.raises(ValueError, match="background"):
        parse_confusion_md("| x | y |\n|---|---|\n| a | 1 |")


def test_confusion_summary_counts() -> None:
    s = confusion_summary(CM, NAMES)
    assert s == {"correct": 9, "wrong_class": 1, "missed": 5, "false_alarms": 13}


def test_tables_show_values_and_skip_missing() -> None:
    r = _run("m", ap50_a=0.25, recall_a=0.5, ap50_b=0.75, recall_b=1.0)
    t = headline_table([r])
    assert "| m | val | 0.5000 | n/a |" in t and "| m | test | 0.6000 |" in t
    assert "| a | 0.2500 | 0.5000 |" in per_class_table([r, _run("agnostic")], NAMES)
    p = protocol_table([r])
    assert "patience 2" in p and "5 / 4" in p and "| 4 |" in p  # best_epoch 3 is shown 1-based


def test_plot_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "cm.png"
    plot_confusion(CM, NAMES, "t", out)
    assert out.read_bytes()[:4] == b"\x89PNG"
