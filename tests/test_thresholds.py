"""Cost-based thresholds: selection optimality, the detection curve, and the protocol guards."""

import json
from pathlib import Path

import numpy as np
import pytest

from defectlens.eval.detection import Detections, GroundTruth, operating_point
from defectlens.eval.threshold_runs import (
    DetectorOutput,
    VisaScores,
    assert_reproduces,
    check_not_a_second_evaluation,
    clean,
    evaluate_all,
    render_report,
    select_all,
    summarize,
    with_notes,
)
from defectlens.eval.thresholds import (
    FrozenEntry,
    ProtocolError,
    candidate_thresholds,
    check_split,
    cost,
    detection_counts,
    detection_flags,
    entries_binary,
    entries_detection,
    evaluate_entry,
    file_sha256,
    flag_counts,
    read_frozen,
    select_binary,
    select_detection,
    trivial_cost,
    wilson_interval,
    write_frozen,
)
from detection_fixtures import random_detections

# ---------------------------------------------------------------------------- protocol guards


@pytest.mark.parametrize(
    ("purpose", "domain", "split"),
    [("selection", "visa", "threshold_calibration"), ("selection", "neu_det", "val"),
     ("evaluation", "visa", "threshold_evaluation"), ("evaluation", "neu_det", "test")],
)
def test_check_split_allows_each_purposes_own_split(purpose: str, domain: str, split: str) -> None:
    check_split(purpose, domain, split)


@pytest.mark.parametrize(
    ("purpose", "domain", "split"),
    [("selection", "visa", "threshold_evaluation"), ("selection", "neu_det", "test"),
     ("selection", "visa", "train"), ("evaluation", "visa", "threshold_calibration"),
     ("evaluation", "neu_det", "val")],
)
def test_check_split_rejects_the_other_split(purpose: str, domain: str, split: str) -> None:
    with pytest.raises(ProtocolError, match="may only read"):
        check_split(purpose, domain, split)


def test_check_split_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="purpose"):
        check_split("tuning", "visa", "threshold_calibration")
    with pytest.raises(ValueError, match="domain"):
        check_split("selection", "mvtec", "val")


def test_selector_rejects_evaluation_data() -> None:
    scores, labels = np.array([0.1, 0.9]), np.array([0, 1])
    with pytest.raises(ProtocolError):
        entries_binary("m", "cat", "threshold_evaluation", scores, labels, [5.0])
    dets, gts = [Detections.empty()], [GroundTruth.of([[0, 0, 5, 5]], [1])]
    with pytest.raises(ProtocolError):
        entries_detection("m", "test", dets, gts, [5.0])


def test_evaluator_rejects_selection_data() -> None:
    entry = FrozenEntry("visa", "m", "cat", 5.0, 0.5, "threshold_calibration", 4, 1, 0, 1, 2, 1.0)
    with pytest.raises(ProtocolError):
        evaluate_entry(entry, "threshold_calibration", scores=[0.1, 0.9], labels=[0, 1])


# ---------------------------------------------------------------------------- binary selection


def test_cost_and_counts() -> None:
    assert cost(fn=2, fp=3, ratio=10) == 23.0
    c = flag_counts([0.1, 0.4, 0.6, 0.9], [0, 1, 0, 1], 0.5)  # flagged: 0.6 (normal), 0.9
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert c.recall == 0.5 and c.precision == 0.5 and c.false_positive_rate == 0.5


def test_candidates_cover_flag_all_and_flag_none() -> None:
    cands = candidate_thresholds([0.2, 0.2, 0.5, 0.9])
    assert len(cands) == 4  # 3 distinct scores -> flag-all, 2 midpoints, flag-none
    assert cands[0] < 0.2 and cands[-1] > 0.9 and np.isclose(cands[1], 0.35)
    assert np.isfinite(cands).all()
    with pytest.raises(ValueError, match="no scores"):
        candidate_thresholds([])


def test_perfectly_separable_scores_cost_nothing() -> None:
    sel = select_binary([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], ratio=10)
    assert sel.cost == 0 and 0.2 < sel.threshold < 0.8


def test_high_ratio_flags_everything_when_classes_overlap() -> None:
    # the lowest-scoring image is anomalous: catching it needs flagging all, which costs 2 FP
    sel = select_binary([0.1, 0.5, 0.6, 0.7, 0.9], [1, 0, 0, 1, 1], ratio=50)
    assert sel.counts.fn == 0 and sel.counts.fp == 2
    sel_low = select_binary([0.1, 0.5, 0.6, 0.7, 0.9], [1, 0, 0, 1, 1], ratio=0.5)
    assert sel_low.counts.fp == 0  # a miss is cheaper than a false alarm: accept the miss


def test_selection_matches_brute_force_over_many_random_sets() -> None:
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(4, 40))
        scores = np.round(rng.normal(size=n) + 1.2 * rng.integers(0, 2, size=n), 1)  # with ties
        labels = rng.integers(0, 2, size=n)
        labels[0], labels[1] = 0, 1
        ratio = float(rng.choice([1.0, 5.0, 10.0, 50.0]))
        sel = select_binary(scores, labels, ratio)
        grid = np.concatenate([np.unique(scores), np.unique(scores) + 0.05, [scores.min() - 1]])
        best = min(flag_counts(scores, labels, float(t)).cost(ratio) for t in grid)
        assert sel.cost == best
        assert flag_counts(scores, labels, sel.threshold) == sel.counts


def test_tie_break_picks_the_median_of_the_tied_candidates() -> None:
    scores, labels = np.array([1.0, 2, 3, 4, 5, 6]), np.array([0, 1, 0, 1, 0, 1])
    cands = candidate_thresholds(scores)
    costs = np.array([flag_counts(scores, labels, float(t)).cost(1.0) for t in cands])
    tied = np.flatnonzero(costs == costs.min())
    assert len(tied) >= 3  # the point of this fixture: several tied optima
    sel = select_binary(scores, labels, 1.0)
    assert sel.threshold == cands[tied[len(tied) // 2]]
    assert sel == select_binary(scores, labels, 1.0)  # repeatable


def test_selection_validates_inputs() -> None:
    with pytest.raises(ValueError, match="labels"):
        select_binary([0.1, 0.2], [0], 5)
    with pytest.raises(ValueError, match="positive"):
        select_binary([0.1, 0.2], [0, 1], 0)


def test_trivial_cost_is_the_better_constant_policy() -> None:
    c = flag_counts([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], 0.5)  # 2 anomalous, 2 normal
    assert trivial_cost(c, ratio=5) == 2  # flag everything: 2 FP < 5 * 2 misses
    assert trivial_cost(c, ratio=0.5) == 1  # flag nothing: 0.5 * 2 misses
    det = detection_counts(
        [Detections.empty()], [GroundTruth.of([[0, 0, 5, 5], [8, 8, 9, 9]], [1, 1])], 0.5
    )
    assert trivial_cost(det, ratio=10) == 20  # keep nothing: 2 boxes missed


def test_wilson_interval() -> None:
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    assert all(np.isnan(v) for v in wilson_interval(0, 0))


# ---------------------------------------------------------------------------- detection selection


def test_detection_curve_matches_operating_point_at_every_candidate() -> None:
    dets, gts = random_detections(np.random.default_rng(3), 30)
    scores, _, _ = detection_flags(dets, gts)
    for ratio in (1.0, 10.0, 50.0):
        sel = select_detection(dets, gts, ratio)
        for t in candidate_thresholds(scores):
            op = operating_point(dets, gts, None, 0.5, float(t))
            assert sel.cost <= ratio * op.fn + op.fp  # nothing beats the selection
        op = operating_point(dets, gts, None, 0.5, sel.threshold)
        assert (sel.counts.tp, sel.counts.fp, sel.counts.fn) == (op.tp, op.fp, op.fn)
        assert detection_counts(dets, gts, sel.threshold) == sel.counts


def test_detection_selection_on_a_hand_checked_example() -> None:
    gts = [GroundTruth.of([[0, 0, 10, 10]], [1]), GroundTruth.of([[0, 0, 10, 10]], [1])]
    dets = [
        Detections.of([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.6], [1, 1]),  # hit + false alarm
        Detections.of([[0, 0, 10, 10]], [0.3], [1]),  # a low-score hit
    ]
    cheap_miss = select_detection(dets, gts, ratio=0.5)
    assert cheap_miss.threshold > 0.6 and cheap_miss.counts.fn == 1 and cheap_miss.counts.fp == 0
    costly_miss = select_detection(dets, gts, ratio=50)
    assert costly_miss.threshold < 0.3 and costly_miss.counts.fn == 0 and costly_miss.counts.fp == 1


def test_detection_selection_without_any_detection_raises() -> None:
    with pytest.raises(ValueError, match="no scores"):
        select_detection([Detections.empty()], [GroundTruth.of([[0, 0, 5, 5]], [1])], 5.0)


def test_wrong_class_counts_as_fp_and_fn() -> None:
    gts = [GroundTruth.of([[0, 0, 10, 10]], [1])]
    dets = [Detections.of([[0, 0, 10, 10]], [0.9], [2])]
    c = detection_counts(dets, gts, 0.5)
    assert (c.tp, c.fp, c.fn) == (0, 1, 1)


# ---------------------------------------------------------------------------- the frozen file


def _entries() -> list[FrozenEntry]:
    return entries_binary("m", "cat", "threshold_calibration", [0.1, 0.2, 0.8, 0.9],
                          [0, 0, 1, 1], [5.0, 10.0])


def test_frozen_round_trip_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    sha = write_frozen(path, _entries(), {"created": "now"})
    assert sha == file_sha256(path)
    entries, meta = read_frozen(path)
    assert entries == _entries() and meta["created"] == "now"


def test_frozen_file_edited_by_hand_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    write_frozen(path, _entries(), {})
    doc = json.loads(path.read_text())
    doc["entries"][0]["threshold"] = 0.123  # a tuned threshold
    path.write_text(json.dumps(doc))
    with pytest.raises(ProtocolError, match="edited"):
        read_frozen(path)


def test_missing_frozen_file_is_a_protocol_error(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="select_thresholds"):
        read_frozen(tmp_path / "nope.json")


def test_frozen_file_cannot_hold_evaluation_split_selections(tmp_path: Path) -> None:
    bad = FrozenEntry("visa", "m", "cat", 5.0, 0.5, "threshold_evaluation", 4, 1, 0, 1, 2, 1.0)
    with pytest.raises(ProtocolError):
        write_frozen(tmp_path / "f.json", [bad], {})


def test_second_evaluation_of_changed_thresholds_is_blocked(tmp_path: Path) -> None:
    results = tmp_path / "threshold_evaluation.json"
    check_not_a_second_evaluation(results, "aaa")  # nothing evaluated yet: fine
    results.write_text(json.dumps({"frozen_file_sha256": "aaa"}))
    check_not_a_second_evaluation(results, "aaa")  # the same frozen file: a repeat, allowed
    with pytest.raises(ProtocolError, match="breach"):
        check_not_a_second_evaluation(results, "bbb")


def test_assert_reproduces() -> None:
    assert assert_reproduces("x", [1.0, 2.0], [1.00001, 2.0]) < 1e-4
    with pytest.raises(ProtocolError, match="does not reproduce"):
        assert_reproduces("x", [1.0, 2.0], [1.0, 2.5])
    with pytest.raises(ProtocolError):
        assert_reproduces("x", [1.0], [1.0, 2.0])


def test_clean_makes_json_safe_values() -> None:
    out = clean({"a": float("nan"), "b": (1, 2), "c": np.float32(0.5), "d": [float("inf")]})
    assert out == {"a": None, "b": [1, 2], "c": 0.5, "d": [None]}
    json.dumps(out, allow_nan=False)


# ---------------------------------------------------------------------------- the whole flow


class Spy:
    """Fake model providers that record which split each call asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.rng = np.random.default_rng(0)

    def visa(self, model: str, category: str, split: str, purpose: str) -> VisaScores:
        self.calls.append((purpose, "visa", split))
        labels = np.array([0, 1] * 10)
        scores = (self.rng.normal(size=20) + 2.0 * labels).astype(np.float32)
        return VisaScores([f"{i}.jpg" for i in range(20)], scores, labels.astype(np.int64))

    def detect(self, run: str, split: str, purpose: str) -> DetectorOutput:
        self.calls.append((purpose, "neu_det", split))
        dets, gts = random_detections(self.rng, 12)
        metrics = {"map50": 0.5, "map50_95": 0.3, "recall": 0.6, "precision": 0.7}
        return DetectorOutput(dets, gts, [f"{i}.jpg" for i in range(12)], metrics)


def test_selection_reads_only_selection_splits_and_evaluation_only_evaluation_splits(
    tmp_path: Path,
) -> None:
    spy = Spy()
    entries = select_all(spy.visa, spy.detect, models=["m1", "m2"], categories=["c1", "c2"],
                         detectors=["d1"], ratios=[5.0, 10.0], iou=0.5)
    assert len(entries) == (2 * 2 + 1) * 2
    assert {(p, d, s) for p, d, s in spy.calls} == {
        ("selection", "visa", "threshold_calibration"), ("selection", "neu_det", "val")}

    path = tmp_path / "frozen.json"
    sha = write_frozen(path, entries, {"created": "t", "git_commit": "g", "iou": 0.5})
    frozen, meta = read_frozen(path)
    spy.calls.clear()
    rows, context = evaluate_all(frozen, spy.visa, spy.detect, 0.5)
    assert {(p, d, s) for p, d, s in spy.calls} == {
        ("evaluation", "visa", "threshold_evaluation"), ("evaluation", "neu_det", "test")}
    assert len(spy.calls) == 4 + 1  # each (model, category) is scored once, not once per ratio
    assert len(rows) == len(entries) and all(r["cost"] >= 0 for r in rows)
    assert set(context) == {"m1/c1", "m1/c2", "m2/c1", "m2/c2", "d1/all"}

    text = render_report(frozen, rows, context, sha, meta)
    assert sha in text and "## VisA, fn_fp_ratio = 5" in text
    assert "## NEU-DET, fn_fp_ratio = 10" in text
    assert "m2" in text and "d1" in text and "Protocol breaches" in text


def test_evaluating_the_selection_data_reproduces_the_stored_counts() -> None:
    """The evaluator's code path gives exactly the selector's counts on the same data."""
    rng = np.random.default_rng(5)
    scores = rng.normal(size=30).astype(np.float32)
    labels = np.array([0, 1] * 15)
    for e in entries_binary("m", "c", "threshold_calibration", scores, labels, [5.0, 50.0]):
        c = flag_counts(scores, labels, e.threshold)
        assert (c.tp, c.fp, c.fn, c.tn) == (e.tp, e.fp, e.fn, e.tn)


def test_summary_counts_and_notes_survive_a_json_round_trip(tmp_path: Path) -> None:
    spy = Spy()
    entries = select_all(spy.visa, spy.detect, models=["m"], categories=["c"], detectors=["d"],
                         ratios=[10.0], iou=0.5)
    rows, context = evaluate_all(entries, spy.visa, spy.detect, 0.5)
    s = summarize(entries, rows)
    assert s["visa_entries"] == 1 and s["neu_entries"] == 1
    assert s["visa_beat_trivial"] + s["visa_tie_trivial"] + s["visa_worse_than_trivial"] == 1
    # as scripts/render_thresholds.py: rows come back from JSON (NaN is null, tuples are lists)
    saved = json.loads(json.dumps(clean({"rows": rows, "context": context}), allow_nan=False))
    text = render_report(entries, saved["rows"], saved["context"], "abc", {"iou": 0.5})
    assert "## Summary" in text and "## VisA, fn_fp_ratio = 10" in text
    notes = tmp_path / "notes.md"
    assert "TODO" in with_notes(text, notes)
    notes.write_text("A hand-written observation.\n")
    assert with_notes(text, notes).rstrip().endswith("A hand-written observation.")
