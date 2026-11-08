"""Leave-one-defect-type-out protocol: splits, operating points, counts and the report."""

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from defectlens.eval.unseen import (
    DETECTOR,
    TypeSplit,
    check_same_recipe,
    choose_thresholds,
    evaluate_thresholds,
    make_type_split,
    render_report,
    threshold_at_fpr,
    type_counts,
)

TYPES: dict[str, tuple[str, ...]] = {}
CAL = [f"c{i:02d}" for i in range(12)]
EVAL = [f"e{i:02d}" for i in range(12)]
for name, t in zip(
    CAL, [(), (), (), (), ("bent",), ("melt",), ("melt",), ("missing",), ("scratch",),
          ("melt", "scratch"), ("bent",), ("missing",)], strict=True):
    TYPES[name] = t
for name, t in zip(
    EVAL, [(), (), (), (), ("bent",), ("melt",), ("melt",), ("missing",), ("scratch",),
           ("melt", "missing"), ("bent",), ("scratch",)], strict=True):
    TYPES[name] = t


def split(holdout: str | None) -> TypeSplit:
    return make_type_split("cat", CAL, EVAL, TYPES, holdout)


def test_split_roles_for_a_holdout() -> None:
    s = split("melt")
    assert s.classes == ("bent", "missing", "scratch")
    # calibration, exactly one type, not melt: the multi-type melt+scratch image is out too
    assert s.train_anomalies == ("c04", "c07", "c08", "c10", "c11")
    assert s.cal_normals == ("c00", "c01", "c02", "c03")
    assert s.eval_normals == ("e00", "e01", "e02", "e03")
    assert s.unseen_eval == ("e05", "e06", "e09")  # includes the multi-type image with melt
    assert s.seen_eval == ("e04", "e07", "e08", "e10", "e11")
    assert s.unseen_all == ("c05", "c06", "c09", "e05", "e06", "e09")


def test_roles_are_disjoint_and_cover_the_halves() -> None:
    for h in (None, "bent", "melt", "missing", "scratch"):
        s = split(h)
        assert set(s.train_anomalies) <= set(CAL)  # never an evaluation image
        assert not set(s.train_anomalies) & set(s.unseen_all)  # held-out type never trained on
        assert not set(s.unseen_eval) & set(s.seen_eval)
        evaluation = set(s.eval_normals) | set(s.unseen_eval) | set(s.seen_eval)
        assert evaluation == set(EVAL)
        assert h is None or all(h in TYPES[p] for p in s.unseen_all)
        assert h is None or not any(h in TYPES[p] for p in s.train_anomalies)


def test_control_has_no_unseen_images_and_all_classes() -> None:
    s = split(None)
    assert s.classes == ("bent", "melt", "missing", "scratch")
    assert s.unseen_eval == () and s.unseen_all == ()
    assert len(s.seen_eval) == 8
    assert len(s.train_anomalies) == 7  # every single-type calibration anomaly


def test_split_errors() -> None:
    with pytest.raises(ValueError, match="not a defect type"):
        split("scrach")
    with pytest.raises(ValueError, match="overlap"):
        make_type_split("cat", CAL, [CAL[0], *EVAL], TYPES, None)
    with pytest.raises(ValueError, match="no type label"):
        make_type_split("cat", [*CAL, "zzz"], EVAL, TYPES, None)


def test_type_counts_count_multi_type_images_under_each_type() -> None:
    assert type_counts(TYPES, CAL) == {"bent": 2, "melt": 3, "missing": 2, "scratch": 2}
    assert type_counts(TYPES, []) == {}


@pytest.mark.parametrize(
    ("scores", "max_fpr", "expected_flagged"),
    [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.0, 0),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.2, 2),  # lowest threshold flagging <= 2 of 10
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.25, 2),  # 2.5 normals are not allowed: round down
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1.0, 10),
        ([1, 1, 1, 5, 5, 9], 0.5, 3),  # ties move together: 3 of 6 flagged (9, 5, 5)
        ([1, 1, 1, 5, 5, 9], 0.4, 1),  # flagging both 5s would be 3 of 6; only 9 fits
    ],
)
def test_threshold_at_fpr_is_the_lowest_allowed_threshold(
    scores: list[float], max_fpr: float, expected_flagged: int
) -> None:
    t = threshold_at_fpr(scores, max_fpr)
    s = np.asarray(scores, dtype=float)
    assert int((s >= t).sum()) == expected_flagged
    assert (s >= t).mean() <= max_fpr
    assert threshold_at_fpr(scores, max_fpr) == t  # deterministic


def test_threshold_at_fpr_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="no normal scores"):
        threshold_at_fpr([], 0.1)
    with pytest.raises(ValueError, match="max_fpr"):
        threshold_at_fpr([1.0], 1.5)


class OnlyCalibrationNormals(Mapping[str, float]):
    """A score table that raises if anything but the calibration normals is read."""

    def __init__(self, allowed: Mapping[str, float]) -> None:
        self.allowed = dict(allowed)

    def __getitem__(self, key: str) -> float:
        if key not in self.allowed:
            msg = f"threshold selection read {key!r}"
            raise AssertionError(msg)
        return self.allowed[key]

    def __iter__(self) -> Any:
        return iter(self.allowed)

    def __len__(self) -> int:
        return len(self.allowed)


def test_thresholds_depend_on_calibration_normals_only() -> None:
    s = split("melt")
    rng = np.random.default_rng(0)
    full = {p: float(rng.random()) for p in (*CAL, *EVAL)}
    only = OnlyCalibrationNormals({p: full[p] for p in s.cal_normals})
    got = choose_thresholds(s, {"m": only}, [0.25, 0.5])  # raises if any other image is read
    assert got == choose_thresholds(s, {"m": full}, [0.25, 0.5])
    # changing every other score (anomalies of both halves, evaluation normals) changes nothing
    changed = {p: (v if p in s.cal_normals else 1.0 - v) for p, v in full.items()}
    assert choose_thresholds(s, {"m": changed}, [0.25, 0.5]) == got


def _scores(
    s: TypeSplit, *, unseen: float, seen: float, normal: float = 0.0
) -> dict[str, float]:
    out = dict.fromkeys(s.cal_normals + s.eval_normals, normal)
    out.update(dict.fromkeys(s.unseen_eval + s.unseen_all, unseen))
    out.update(dict.fromkeys(s.seen_eval, seen))
    return out


def test_evaluate_thresholds_counts() -> None:
    s = split("melt")
    # normals score 0; held-out anomalies 1 (all caught); seen anomalies 0 (all missed), except one
    sc = _scores(s, unseen=1.0, seen=0.0)
    sc["e04"] = 1.0
    sc["e00"] = 1.0  # one false alarm on the evaluation normals
    rows = evaluate_thresholds(s, {"m": sc}, choose_thresholds(s, {"m": sc}, [0.0]))
    (r,) = rows
    assert r["holdout"] == "melt" and r["max_fpr"] == 0.0
    assert (r["unseen_tp"], r["n_unseen"]) == (3, 3)
    assert (r["seen_tp"], r["n_seen"]) == (1, 5)
    assert (r["cal_fp"], r["n_cal_normals"]) == (0, 4)
    assert (r["eval_fp"], r["n_eval_normals"]) == (1, 4)
    assert r["n_unseen_all"] == 6
    # 6 unseen images at 1.0 against 8 normals at 0 except e00 at 1.0 (a tie counts half)
    assert r["auroc_unseen"] == pytest.approx((6 * 7 + 6 * 0.5) / (6 * 8))


def test_control_run_has_nan_unseen_auroc() -> None:
    s = split(None)
    sc = _scores(s, unseen=1.0, seen=1.0)
    (r,) = evaluate_thresholds(s, {"m": sc}, {"m": {0.1: 0.5}})
    assert r["holdout"] == "none" and r["n_unseen"] == 0 and r["n_unseen_all"] == 0
    assert np.isnan(r["auroc_unseen"]) and r["auroc_seen"] == 1.0


def _result(holdout: str | None, **kw: Any) -> dict[str, Any]:
    s = split(holdout)
    scores = {
        DETECTOR: _scores(s, unseen=0.05, seen=0.9, normal=0.1),
        "patchcore": _scores(s, unseen=0.8, seen=0.8, normal=0.3),
    }
    th = choose_thresholds(s, scores, [0.05, 0.1])
    return {
        "holdout": holdout or "none", "n_cal_normals": 4, "n_eval_normals": 4,
        "n_train_normals_available": 904,
        "type_counts": {"calibration": type_counts(TYPES, CAL),
                        "evaluation": type_counts(TYPES, EVAL),
                        "both": type_counts(TYPES, [*CAL, *EVAL])},
        "detector": {"classes": list(s.classes), "n_train_anomalies": len(s.train_anomalies),
                     "n_train_normals": 100, "epochs": 20, "iters_per_epoch": 60, "seconds": 300.0,
                     "peak_vram_alloc_mib": 2500.0, "peak_vram_reserved_mib": 3000.0,
                     "loss": [1.0, 0.5], "peak_ram_mib": 5000.0},
        "rows": evaluate_thresholds(s, scores, th),
        "mlflow_run_id": "abc123",
        "config": {"unseen": {"holdout": holdout, "epochs": 20, "force": False, "smoke": False},
                   "model": {"lr": 0.0025}, "preprocess": {"size": 640}, "seed": 42,
                   "trainer": {"amp": True}},
        **kw,
    }


def test_render_report_tables_and_stub() -> None:
    stub = render_report([])
    assert "TODO" in stub and "## Protocol" in stub and "| melt |" not in stub
    text = render_report([_result("melt"), _result("bent"), _result(None)])
    for heading in ("## Images per defect type", "## Detector training",
                    "## Recall on the held-out type, false-alarm budget 5%",
                    "## Recall on the other (seen) types, budget 10%",
                    "## Threshold-free: image AUROC"):
        assert heading in text
    assert "3/3 = 1.00" in text  # patchcore catches every unseen melt image in this fake
    assert "0/3 = 0.00" in text  # the fake detector catches none
    assert "`abc123`" in text and "| none |" in text


def test_check_same_recipe() -> None:
    a, b = _result("melt"), _result("bent")
    check_same_recipe([a, b])  # holdout differs, nothing else: fine
    check_same_recipe([])
    c = copy.deepcopy(b)
    c["config"]["unseen"]["epochs"] = 30
    with pytest.raises(ValueError, match="different recipes"):
        check_same_recipe([a, c])
    d = copy.deepcopy(b)
    d["config"]["model"]["lr"] = 0.01
    with pytest.raises(ValueError, match="different recipes"):
        check_same_recipe([a, d])
