"""EXPLORATORY follow-up to Phase 4b: the same detector with a defect type trained on vs held out.

This is **outside the approved Phase 4b protocol** (five runs, written down before they were made).
It runs no model and reads no data: it only re-reads the per-image scores already saved in
``reports/unseen/pcb1_*.json``. The evaluation half has now been read a third time, so nothing
here may feed back into a threshold, a recipe or a model choice.

Question: for a defect type T, is the Faster R-CNN's recall on the evaluation images that contain
T lower when T was held out of its training set than in the control run, where T was trained on?

Analysis, fixed before any number was computed:

* Images: the evaluation-half anomalies containing T (``split.unseen_eval`` of the run that held T
  out). The control run saw no evaluation image, so both detectors are scored on the same images.
* Operating point: each run's own threshold from the saved result (lowest threshold with at most
  5% or 10% false alarms on the calibration normals), exactly as in Phase 4b.
* Primary: image-level recall, paired by image (caught in the held-out run, caught in the control
  run), with an exact two-sided McNemar test (binomial on the discordant pairs). One test per
  type and budget, **not corrected for the 8 tests**, which are not independent either.
* Secondary: the same restricted to images that carry only the type T. For a multi-type image the
  held-out detector still knows the other type, so such an image does not test novelty cleanly.
  The types of an image are read from which runs list it as ``unseen_eval``.
* Threshold-free: image AUROC of those evaluation images against all 100 test normals (the
  calibration and evaluation halves) in both runs. This is not the Phase 4b AUROC: that one uses
  both halves, and the control detector trained on the calibration-half images.
* Negative control: the anomaly models are the same checkpoints in every run, so their scores for
  an image must be identical across runs and their paired difference must be exactly zero. The
  largest score difference across runs is reported.

What the comparison can not separate (stated in the report): the two detectors differ in the
training set size and composition (23 to 47 anomalous images; held-out melt loses half of them),
in the random state of one 20-epoch run each, and in the threshold, besides the one type.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import binomtest

from defectlens.eval.anomaly import auroc
from defectlens.eval.thresholds import wilson_interval
from defectlens.eval.unseen import DETECTOR, MODEL_ORDER, NO_HOLDOUT

Result = Mapping[str, Any]


@dataclass(frozen=True)
class Paired:
    """Recall of one model on the evaluation images containing ``holdout``: held out vs control."""

    model: str
    holdout: str
    max_fpr: float
    n: int
    both: int  # caught in the held-out run and in the control run
    held_only: int  # caught only when the type was held out
    control_only: int  # caught only when the type was trained on
    neither: int
    p_exact: float  # two-sided exact McNemar; 1.0 when there is no discordant pair
    n_single: int  # images carrying only this type
    held_single: int  # caught in the held-out run, among those
    control_single: int  # caught in the control run, among those
    auroc_held: float
    auroc_control: float

    @property
    def held_tp(self) -> int:
        return self.both + self.held_only

    @property
    def control_tp(self) -> int:
        return self.both + self.control_only


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts ``b`` and ``c``."""
    if b < 0 or c < 0:
        msg = f"discordant counts must be >= 0, got {b}, {c}"
        raise ValueError(msg)
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def image_types(runs: Mapping[str, Result]) -> dict[str, frozenset[str]]:
    """Defect types of every evaluation image that carries one, from the held-out runs' lists."""
    out: dict[str, set[str]] = {}
    for holdout, r in runs.items():
        if holdout == NO_HOLDOUT:
            continue
        for p in r["split"]["unseen_eval"]:
            out.setdefault(p, set()).add(holdout)
    return {p: frozenset(t) for p, t in out.items()}


def _threshold(r: Result, model: str, max_fpr: float) -> float:
    return float(r["thresholds"][model][str(max_fpr)])


def _auc(r: Result, model: str, positives: Sequence[str]) -> float:
    sc = r["scores"][model]
    normals = [*r["split"]["cal_normals"], *r["split"]["eval_normals"]]
    pos = np.array([sc[p] for p in positives], dtype=np.float64)
    neg = np.array([sc[p] for p in normals], dtype=np.float64)
    return auroc(np.concatenate([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))])


def paired_comparison(
    runs: Mapping[str, Result],
    models: Sequence[str] | None = None,
    max_fprs: Sequence[float] | None = None,
) -> list[Paired]:
    """One ``Paired`` per (held-out type, model, budget); needs the control run ``none``.

    Models and budgets default to those saved in the control run's thresholds.
    """
    if NO_HOLDOUT not in runs:
        msg = "the control run (holdout=none) is missing"
        raise ValueError(msg)
    control = runs[NO_HOLDOUT]
    saved = control["thresholds"]
    models = [m for m in MODEL_ORDER if m in saved] if models is None else models
    max_fprs = (
        sorted({float(a) for m in models for a in saved[m]}) if max_fprs is None else max_fprs
    )
    types = image_types(runs)
    out: list[Paired] = []
    for holdout in sorted(h for h in runs if h != NO_HOLDOUT):
        held = runs[holdout]
        images = list(held["split"]["unseen_eval"])
        single = [p for p in images if types[p] == frozenset({holdout})]
        for model in models:
            for a in max_fprs:
                th, tc = _threshold(held, model, a), _threshold(control, model, a)
                hs, cs = held["scores"][model], control["scores"][model]

                def caught(
                    paths: Sequence[str], th: float = th, tc: float = tc,
                    hs: Mapping[str, float] = hs, cs: Mapping[str, float] = cs,
                ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
                    return (np.array([hs[p] >= th for p in paths], dtype=bool),
                            np.array([cs[p] >= tc for p in paths], dtype=bool))

                h_all, c_all = caught(images)
                h_one, c_one = caught(single)
                b, c = int((h_all & ~c_all).sum()), int((~h_all & c_all).sum())
                out.append(Paired(
                    model=model, holdout=holdout, max_fpr=a, n=len(images),
                    both=int((h_all & c_all).sum()), held_only=b, control_only=c,
                    neither=int((~h_all & ~c_all).sum()), p_exact=mcnemar_exact(b, c),
                    n_single=len(single), held_single=int(h_one.sum()),
                    control_single=int(c_one.sum()),
                    auroc_held=_auc(held, model, images),
                    auroc_control=_auc(control, model, images),
                ))
    return out


def check_against_saved_rows(runs: Mapping[str, Result], paired: Sequence[Paired]) -> None:
    """The held-out counts recomputed here must equal the Phase 4b rows (a plumbing guard)."""
    for p in paired:
        row = next(
            r for r in runs[p.holdout]["rows"]
            if r["model"] == p.model and r["max_fpr"] == p.max_fpr
        )
        if (row["unseen_tp"], row["n_unseen"]) != (p.held_tp, p.n):
            msg = (f"{p.model}/{p.holdout}/{p.max_fpr}: recomputed {p.held_tp}/{p.n} but the saved "
                   f"row says {row['unseen_tp']}/{row['n_unseen']}")
            raise ValueError(msg)


def max_score_difference(runs: Mapping[str, Result], model: str) -> float:
    """Largest absolute difference of one model's score for the same image across the runs."""
    worst = 0.0
    base = next(iter(runs.values()))["scores"][model]
    for r in runs.values():
        sc = r["scores"][model]
        worst = max(worst, max(abs(sc[p] - base[p]) for p in base))
    return worst


def _rate(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} = {k / n:.2f} [{lo:.2f}, {hi:.2f}]"


def _num(x: float) -> str:
    return "-" if math.isnan(x) else f"{x:.3f}"


def _pct(x: float) -> str:
    return f"{x * 100:g}%"


BANNER = (
    "> **EXPLORATORY. Outside the approved Phase 4b protocol.** Added after the five runs and "
    "after the verdict was written, at the project owner's request, using only the per-image "
    "scores saved in `reports/unseen/pcb1_*.json` (no new run, no data read). The evaluation half "
    "has now been read a third time. Nothing here changes a threshold, a recipe or the Phase 4b "
    "verdict; the p-values are not corrected for the 8 tests and are descriptive."
)


def render_followup(runs: Mapping[str, Result]) -> str:
    """The exploratory report; a stub with a TODO when the control run is not there yet."""
    lines = [
        "# Exploratory: the detector with a type trained on vs held out (follow-up to Phase 4b)",
        "",
        BANNER,
        "",
        "Generated by `scripts/render_unseen_followup.py`; the commentary is in "
        "`reports/unseen_followup_notes.md`.",
        "",
        "## Question and analysis",
        "",
        "For each defect type T: is the Faster R-CNN's recall on the evaluation images that "
        "contain T lower when T was held out of its training set than in the control run (T "
        "trained on)? Both detectors score the same images (no evaluation image is ever in a "
        "training set). Each run uses its own 5% / 10% operating point from the calibration "
        "normals, as in Phase 4b. Tests are exact McNemar on the paired outcomes, two-sided. The "
        "*single-type* columns drop images that also carry another type, which the held-out "
        "detector knows. AUROC here is these evaluation images against the 100 test normals "
        "(not the Phase 4b AUROC, which includes calibration-half images the control detector "
        "trained on).",
        "",
        "**What this can not separate from novelty:** the two detectors also differ in training "
        "set size (23 to 47 anomalous images; held-out melt loses about half), in the random "
        "state of one 20-epoch run each, and in the threshold. Type difficulty is controlled "
        "(same images), but the training-set change is not.",
        "",
    ]
    if NO_HOLDOUT not in runs or len(runs) < 2:
        return "\n".join([*lines, "TODO: needs the control run and at least one held-out run.", ""])

    paired = paired_comparison(runs)
    check_against_saved_rows(runs, paired)
    types = image_types(runs)
    multi = sorted(p for p, t in types.items() if len(t) > 1)

    det = [p for p in paired if p.model == DETECTOR]
    for a in sorted({p.max_fpr for p in paired}):
        lines += [f"## Detector recall, budget {_pct(a)}", "",
                  "| type | images | held out | trained on (control) | held-out only | "
                  "control only | p (exact) | single-type: held out | single-type: control |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for p in (q for q in det if q.max_fpr == a):
            lines.append(
                f"| {p.holdout} | {p.n} | {_rate(p.held_tp, p.n)} | {_rate(p.control_tp, p.n)} | "
                f"{p.held_only} | {p.control_only} | {p.p_exact:.3f} | "
                f"{_rate(p.held_single, p.n_single)} | {_rate(p.control_single, p.n_single)} |"
            )
        lines.append("")

    lines += ["## Detector image AUROC on the same images", "",
              "| type | images | held out | trained on (control) |", "|---|---|---|---|"]
    for p in (q for q in det if q.max_fpr == det[0].max_fpr):
        lines.append(f"| {p.holdout} | {p.n} | {_num(p.auroc_held)} | {_num(p.auroc_control)} |")
    lines += ["", "No AUROC interval was computed; a difference of a few hundredths is not "
              "distinguishable.", ""]

    lines += ["## Anomaly models: negative control", "",
              "The anomaly checkpoints are identical in every run, so their paired difference must "
              "be zero and their scores identical. A non-zero value would mean the pairing is "
              "wrong.", "",
              "| model | largest score difference across runs | held-out-only + control-only "
              "images, all types and budgets |", "|---|---|---|"]
    for m in (m for m in MODEL_ORDER if m != DETECTOR and any(q.model == m for q in paired)):
        disc = sum(q.held_only + q.control_only for q in paired if q.model == m)
        lines.append(f"| {m} | {max_score_difference(runs, m):.3g} | {disc} |")
    lines.append("")

    lines += ["## Types per image", "",
              f"{len(multi)} of {len(types)} evaluation anomalies carry more than one type: "
              + (", ".join(f"`{p.rsplit('/', 1)[-1]}` ({'+'.join(sorted(types[p]))})"
                           for p in multi) or "none") + ".", ""]
    return "\n".join(lines)
