"""Exploratory follow-up to Phase 4b: paired comparison of a type trained on vs held out."""

from typing import Any

import pytest

from defectlens.eval.unseen_followup import (
    check_against_saved_rows,
    image_types,
    max_score_difference,
    mcnemar_exact,
    paired_comparison,
    render_followup,
)

CAL_N = ["cn0", "cn1", "cn2", "cn3"]
EVAL_N = ["en0", "en1", "en2", "en3"]
# eval anomalies: a0..a2 carry bent, a2 and a3 carry melt (a2 has both)
BENT = ["a0", "a1", "a2"]
MELT = ["a2", "a3"]


def run(holdout: str, det: dict[str, float], thr: float, anomaly: float = 0.5) -> dict[str, Any]:
    """A result file: detector scores ``det``, normals at 0.1..0.4, plus one anomaly model."""
    normals = dict(zip(CAL_N + EVAL_N, [0.1, 0.2, 0.3, 0.4] * 2, strict=True))
    unseen = {"bent": BENT, "melt": MELT, "none": []}[holdout]
    split = {"cal_normals": CAL_N, "eval_normals": EVAL_N, "unseen_eval": unseen}
    scores = {
        "faster_rcnn": {**normals, **det},
        "patchcore": {**normals, **dict.fromkeys(det, anomaly)},
    }
    rows = []
    for model in scores:
        t = thr if model == "faster_rcnn" else 0.45
        rows.append({
            "model": model, "max_fpr": 0.05, "threshold": t,
            "unseen_tp": sum(scores[model][p] >= t for p in unseen), "n_unseen": len(unseen),
        })
    thresholds = {"faster_rcnn": {"0.05": thr}, "patchcore": {"0.05": 0.45}}
    return {"holdout": holdout, "split": split, "scores": scores, "thresholds": thresholds,
            "rows": rows}


@pytest.fixture
def runs() -> dict[str, dict[str, Any]]:
    everything = ["a0", "a1", "a2", "a3"]
    return {
        # control: catches everything
        "none": run("none", dict.fromkeys(everything, 0.9), 0.5),
        # bent held out: catches a0 only among bent; melt images a2 (also bent) / a3 irrelevant here
        "bent": run("bent", {"a0": 0.9, "a1": 0.2, "a2": 0.2, "a3": 0.9}, 0.5),
        # melt held out: catches a3, loses a2
        "melt": run("melt", {"a0": 0.9, "a1": 0.9, "a2": 0.2, "a3": 0.9}, 0.5),
    }


def test_mcnemar_exact() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == pytest.approx(1.0)
    assert mcnemar_exact(0, 5) == pytest.approx(2 * 0.5**5)
    assert mcnemar_exact(5, 0) == mcnemar_exact(0, 5)
    with pytest.raises(ValueError, match=">= 0"):
        mcnemar_exact(-1, 2)


def test_image_types_reads_the_overlap_between_runs(runs: dict[str, dict[str, Any]]) -> None:
    t = image_types(runs)
    assert t["a2"] == frozenset({"bent", "melt"})
    assert t["a0"] == frozenset({"bent"})
    assert t["a3"] == frozenset({"melt"})


def test_paired_counts(runs: dict[str, dict[str, Any]]) -> None:
    out = {(p.model, p.holdout): p for p in paired_comparison(runs, ["faster_rcnn"], [0.05])}
    bent = out[("faster_rcnn", "bent")]
    assert (bent.n, bent.held_tp, bent.control_tp) == (3, 1, 3)
    assert (bent.both, bent.held_only, bent.control_only, bent.neither) == (1, 0, 2, 0)
    assert bent.p_exact == pytest.approx(0.5)
    # single-type bent images are a0 and a1 (a2 also carries melt)
    assert (bent.n_single, bent.held_single, bent.control_single) == (2, 1, 2)
    melt = out[("faster_rcnn", "melt")]
    assert (melt.n, melt.held_tp, melt.control_tp, melt.control_only) == (2, 1, 2, 1)
    # control AUROC is perfect (anomalies 0.9 above every normal); held-out loses a2
    assert melt.auroc_control == 1.0
    assert melt.auroc_held < melt.auroc_control
    check_against_saved_rows(runs, list(out.values()))


def test_unchanged_anomaly_model_has_no_discordant_pairs(runs: dict[str, dict[str, Any]]) -> None:
    for p in paired_comparison(runs, ["patchcore"], [0.05]):
        assert p.held_only == 0 and p.control_only == 0 and p.p_exact == 1.0
    assert max_score_difference(runs, "patchcore") == 0.0
    assert max_score_difference(runs, "faster_rcnn") == pytest.approx(0.7)


def test_check_against_saved_rows_catches_plumbing_errors(
    runs: dict[str, dict[str, Any]],
) -> None:
    paired = paired_comparison(runs, ["faster_rcnn"], [0.05])
    runs["bent"]["rows"][0]["unseen_tp"] = 3  # the saved row disagrees with the scores
    with pytest.raises(ValueError, match="recomputed"):
        check_against_saved_rows(runs, paired)


def test_control_run_is_required(runs: dict[str, dict[str, Any]]) -> None:
    del runs["none"]
    with pytest.raises(ValueError, match="control run"):
        paired_comparison(runs, ["faster_rcnn"], [0.05])


def test_render_labels_itself_exploratory_and_stubs_without_control(
    runs: dict[str, dict[str, Any]],
) -> None:
    text = render_followup({"bent": runs["bent"]})
    assert "EXPLORATORY" in text and "TODO" in text
    full = render_followup(runs)
    assert "EXPLORATORY" in full and "Outside the approved Phase 4b protocol" in full
    assert "Anomaly models: negative control" in full
