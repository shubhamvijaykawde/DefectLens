"""Anomaly comparison tables, few-shot statistics and the figure, on synthetic runs."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defectlens.eval import anomaly_compare as ac


def _run(model: str, category: str, name: str | None = None, phase: str = "3a",
         shots: int | None = None, draw: int | None = None, **metrics: float) -> ac.AnomalyRun:
    base = {"image_auroc": 0.5, "pixel_auroc": 0.6, "aupro": 0.7}
    base.update(metrics)
    return ac.AnomalyRun(
        name or f"{model}_{category}", "abcdef012345", model, category, phase, shots, draw, base,
        {"fit_seconds": 10.0, "peak_vram_alloc_mib": 100.0 if model == "patchcore" else 50.0},
    )


def test_full_runs_keeps_only_exactly_named_full_data_runs() -> None:
    runs = [
        _run("patchcore", "a"),
        _run("efficientad", "a", phase="3b"),
        _run("efficientad", "a", name="efficientad_a_smoke500", phase="3b"),
        _run("anomaly_dino", "a", name="anomaly_dino_a_5shot_d0", phase="3c", shots=5, draw=0),
    ]
    full = ac.full_runs(runs)
    assert set(full) == {("patchcore", "a"), ("efficientad", "a")}
    assert full["efficientad", "a"].name == "efficientad_a"


def test_mean_std_uses_sample_std_and_nan_for_one_value() -> None:
    m, s = ac.mean_std([0.8, 0.9, 1.0])
    assert m == pytest.approx(0.9) and s == pytest.approx(0.1)
    assert math.isnan(ac.mean_std([0.7])[1])


def test_full_and_cost_tables() -> None:
    full = ac.full_runs([
        _run("patchcore", "a", image_auroc=0.9), _run("patchcore", "b", image_auroc=0.7),
        _run("anomaly_dino", "a", image_auroc=0.8),
    ])
    table = ac.full_table(full, ["patchcore", "anomaly_dino"], ["a", "b"], "image_auroc")
    rows = table.splitlines()
    assert rows[0] == "| category | PatchCore | AnomalyDINO |"
    assert rows[2] == "| a | 0.9000 | 0.8000 |"
    assert rows[3] == "| b | 0.7000 | n/a |"  # missing run shown, not invented
    assert "**0.8000**" in rows[4]
    cost = ac.cost_table(full, ["patchcore", "anomaly_dino"], ["a", "b"])
    assert "| PatchCore | 10 | 10 | 100 |" in cost and "| AnomalyDINO | 10 | n/a | 50 |" in cost


def _few(values: dict[tuple[str, int, int], float]) -> list[ac.AnomalyRun]:
    return [
        _run("anomaly_dino", c, name=f"x_{c}_{k}_{d}", phase="3c", shots=k, draw=d, image_auroc=v)
        for (c, k, d), v in values.items()
    ]


def test_few_shot_stats_per_category_and_mean_over_draw_averages() -> None:
    runs = _few({
        ("a", 1, 0): 0.6, ("a", 1, 1): 0.8, ("a", 5, 0): 0.9,
        ("b", 1, 0): 0.4, ("b", 1, 1): 0.6, ("b", 5, 0): 0.7, ("b", 5, 1): 0.9,
    })
    values = ac.few_shot_values(runs, "anomaly_dino", "image_auroc")
    stats = ac.few_shot_stats(values, ["a", "b"], [1, 5, 10])
    assert stats["a"][1][0] == pytest.approx(0.7) and stats["a"][1][2] == 2
    assert stats["a"][1][1] == pytest.approx(math.sqrt(0.02))
    assert math.isnan(stats["a"][5][1]) and stats["a"][5][2] == 1
    assert 10 not in stats["a"]  # no runs, no entry
    # mean panel: per-draw category averages (0.5, 0.7), then mean and std over draws
    assert stats["mean"][1][0] == pytest.approx(0.6)
    assert stats["mean"][1][1] == pytest.approx(math.sqrt(0.02))
    # 5 shots: only draw 0 exists in both categories, so n=1
    assert stats["mean"][5][2] == 1 and stats["mean"][5][0] == pytest.approx(0.8)
    assert 10 not in stats["mean"]


def test_few_shot_values_ignores_other_models_and_phases() -> None:
    runs = _few({("a", 1, 0): 0.6}) + [
        _run("patchcore", "a", phase="3c", shots=1, draw=0, image_auroc=0.1),
        _run("anomaly_dino", "a"),
    ]
    assert ac.few_shot_values(runs, "anomaly_dino", "image_auroc") == {"a": {1: {0: 0.6}}}


def test_few_shot_table_text() -> None:
    stats = {"a": {1: (0.7, 0.1, 5)}, "mean": {1: (0.7, float("nan"), 1)}}
    table = ac.few_shot_table(stats, {"a": 0.95}, [1, 5], ["a", "mean"])
    rows = table.splitlines()
    assert rows[0] == "| category | 1-shot | 5-shot | full data |"
    assert rows[2] == "| a | 0.7000 ± 0.1000 (n=5) | n/a | 0.9500 |"
    assert rows[3] == "| mean | 0.7000 ± n/a (n=1) | n/a | n/a |"


def test_plot_writes_a_png(tmp_path: Path) -> None:
    cats = ["a", "b"]
    grid = [(c, k, d) for c in cats for k in (1, 5) for d in (0, 1)]
    runs = _few({(c, k, d): 0.5 + 0.03 * k + 0.01 * d for c, k, d in grid})
    values = ac.few_shot_values(runs, "anomaly_dino", "image_auroc")
    stats = ac.few_shot_stats(values, cats, [1, 5])
    refs = {"anomaly_dino": {"a": 0.9, "b": 0.85, "mean": 0.875},
            "patchcore": {"a": 0.92, "b": 0.8, "mean": 0.86}}
    out = tmp_path / "fig" / "curve.png"
    ac.plot_few_shot(values, stats, refs, cats, [1, 5], out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_load_runs_keeps_latest_per_name_and_parses_tags() -> None:
    def rec(name: str, run_id: str, auroc: float, **tags: str) -> SimpleNamespace:
        t = {"model": "anomaly_dino", "category": "a", "phase": "3c", "eval_split": "x", **tags}
        return SimpleNamespace(
            info=SimpleNamespace(run_name=name, run_id=run_id),
            data=SimpleNamespace(tags=t, metrics={"calib/image_auroc": auroc, "fit_seconds": 2.0}),
        )

    recs = [  # newest first, as MLflow returns them
        rec("n1", "new", 0.9, shots="5", draw="2"), rec("n1", "old", 0.1, shots="5", draw="2"),
        rec("full", "f", 0.8, phase="3a"),
    ]
    client = SimpleNamespace(search_runs=lambda *a, **k: recs)
    runs = {r.name: r for r in ac.load_runs(client, "1")}
    assert set(runs) == {"n1", "full"} and runs["n1"].run_id == "new"
    assert (runs["n1"].shots, runs["n1"].draw) == (5, 2)
    assert runs["full"].shots is None and runs["full"].draw is None
    assert runs["n1"].metrics == {"image_auroc": 0.9} and runs["n1"].scalars == {"fit_seconds": 2.0}
    assert np.isfinite(runs["n1"].metrics["image_auroc"])
