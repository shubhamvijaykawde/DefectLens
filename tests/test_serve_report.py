"""The TF32 rule and the Phase 6a report (reads the committed result files, runs no model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from defectlens.export.serve_report import (
    decide_all,
    flip_total,
    latency_saving,
    render_serve_models,
    tf32_decision,
)
from defectlens.serve.config import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_flip_total_sums_images_over_the_ratios() -> None:
    assert flip_total({"by_ratio": {"5": {"flips": 1}, "10": {"flips": 2}}}) == 3
    assert flip_total({"by_ratio": {"5": {"images_kept_count_differs": 4}}}) == 4


def test_latency_saving_reads_the_ort_cuda_rows() -> None:
    rows = [
        {"model": "efficientad", "variant": "ort_cuda_tf32_off", "latency": {"p50_ms": 10.0}},
        {"model": "efficientad", "variant": "ort_cuda_tf32_on", "latency": {"p50_ms": 8.0}},
        {"model": "efficientad", "variant": "torch_fp32", "latency": {"p50_ms": 1.0}},
    ]
    assert latency_saving(rows, "efficientad") == (10.0, 8.0, pytest.approx(0.2))


@pytest.mark.parametrize(("flips", "saving", "expected"),
                         [(0, 0.30, True), (0, 0.19, False), (1, 0.50, False), (0, 0.20, True)])
def test_tf32_rule(flips: int, saving: float, expected: bool) -> None:
    assert tf32_decision(flips, saving) is expected


def test_the_served_tf32_settings_follow_the_rule_on_the_committed_results() -> None:
    settings = load_settings(ROOT / "configs/serve/default.yaml")
    consistency = json.loads((ROOT / "reports/serve_threshold_consistency.json").read_text())
    bench = json.loads((ROOT / "reports/benchmark.json").read_text())["rows"]
    decided = decide_all(settings, consistency, bench)
    assert set(decided) == set(settings.products), "every served product has a measured result"
    for name, d in decided.items():
        assert d["tf32"] == d["configured"], name


def test_report_renders_every_section() -> None:
    settings = load_settings(ROOT / "configs/serve/default.yaml")
    consistency = json.loads((ROOT / "reports/serve_threshold_consistency.json").read_text())
    bench = json.loads((ROOT / "reports/benchmark.json").read_text())["rows"]
    text = render_serve_models(settings, consistency, bench)
    for heading in ("## Served products", "## TF32 rule applied", "VisA products", "detectors"):
        assert heading in text
    assert "visa_pcb1" in text and "steel (faster_rcnn_aug)" in text
