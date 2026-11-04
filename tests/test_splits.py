import json
from pathlib import Path

import pytest

from defectlens.data.splits import (
    assert_disjoint,
    load_splits,
    save_splits,
    split_hash,
    stratified_split,
)
from defectlens.data.visa_splits import make_visa_splits

ITEMS = [(f"c{c}_{i}", f"c{c}") for c in range(3) for i in range(20)]
FR = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_sizes_per_stratum_and_disjoint() -> None:
    parts = stratified_split(ITEMS, FR, seed=1)
    assert [len(parts[k]) for k in ("train", "val", "test")] == [42, 9, 9]
    assert_disjoint(parts)
    for k in parts:
        for c in range(3):
            n = sum(i.startswith(f"c{c}_") for i in parts[k])
            assert n == {"train": 14, "val": 3, "test": 3}[k]


def test_deterministic_and_order_independent() -> None:
    a = stratified_split(ITEMS, FR, seed=1)
    b = stratified_split(list(reversed(ITEMS)), FR, seed=1)
    assert a == b
    assert a != stratified_split(ITEMS, FR, seed=2)


def test_bad_inputs() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        stratified_split(ITEMS, {"a": 0.5, "b": 0.6}, 0)
    with pytest.raises(ValueError, match="duplicate"):
        stratified_split([("x", "a"), ("x", "a")], {"a": 0.5, "b": 0.5}, 0)
    with pytest.raises(ValueError, match="both"):
        assert_disjoint({"a": ["x"], "b": ["x"]})


def test_save_load_and_tamper_detection(tmp_path: Path) -> None:
    parts = stratified_split(ITEMS, FR, seed=1)
    p = tmp_path / "s.json"
    digest = save_splits(parts, {"seed": 1}, p)
    assert digest == split_hash(parts) == load_splits(p)["meta"]["split_hash"]
    doc = json.loads(p.read_text())
    doc["test"].append("sneaky")
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_splits(p)


def _row(split: str, label: str, name: str) -> dict[str, str]:
    return {"object": "x", "split": split, "label": label, "image": f"x/{name}"}


def _rows() -> list[dict[str, str]]:
    rows = [_row("train", "normal", f"n{i}") for i in range(6)]
    rows += [_row("test", "normal", f"tn{i}") for i in range(5)]
    rows += [_row("test", "anomaly", f"ta{i}") for i in range(4)]
    return rows


def test_visa_halves_stratified_disjoint_and_calibration_gets_extra() -> None:
    parts, labels = make_visa_splits(_rows(), seed=42)
    assert_disjoint(parts)
    assert len(parts["train"]) == 6
    cal, ev = parts["threshold_calibration"], parts["threshold_evaluation"]
    count = lambda ids, lab: sum(labels[i] == lab for i in ids)  # noqa: E731
    assert (count(cal, "normal"), count(ev, "normal")) == (3, 2)
    assert (count(cal, "anomaly"), count(ev, "anomaly")) == (2, 2)
    assert parts == make_visa_splits(_rows(), seed=42)[0]


def test_visa_rejects_anomalous_train() -> None:
    rows = _rows()
    rows[0]["label"] = "anomaly"
    with pytest.raises(ValueError, match="non-normal"):
        make_visa_splits(rows, 0)
