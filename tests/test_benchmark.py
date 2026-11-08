"""Benchmark helpers: summary statistics, timing loop, memory sampler, variants."""

import itertools
import time

import pytest

from defectlens.export.benchmark import (
    MemorySampler,
    Variant,
    speedup,
    summarize,
    time_calls,
    variants,
)


def test_summarize_percentiles_and_throughput() -> None:
    s = summarize(list(range(1, 101)))  # 1..100 ms
    assert s["p50_ms"] == pytest.approx(50.5) and s["p95_ms"] == pytest.approx(95.05)
    assert s["mean_ms"] == pytest.approx(50.5)
    assert s["throughput_per_s"] == pytest.approx(1000 / 50.5)
    assert s["min_ms"] == 1 and s["max_ms"] == 100 and s["n"] == 100
    with pytest.raises(ValueError, match="no timings"):
        summarize([])


def test_time_calls_runs_warmup_untimed_and_times_the_rest() -> None:
    calls: list[int] = []
    times = time_calls(lambda: (calls.append(1), time.sleep(0.002)), warmup=3, runs=5)
    assert len(calls) == 8 and len(times) == 5
    assert all(t >= 1.5 for t in times)
    with pytest.raises(ValueError, match="runs"):
        time_calls(lambda: None, warmup=0, runs=0)


def test_memory_sampler_keeps_the_maximum_including_the_end() -> None:
    readings = itertools.chain([10.0, 30.0, 20.0], itertools.repeat(25.0))
    with MemorySampler(lambda: next(readings), period_s=0.001) as s:
        time.sleep(0.02)
    assert s.peak == 30.0


def test_variants_and_labels_round_trip() -> None:
    plan = variants(["cuda", "cpu"], [False, True], ["fp32", "fp16_amp"])
    labels = [v.label for v in plan]
    assert labels == ["ort_cuda_tf32_off", "ort_cuda_tf32_on", "ort_cpu", "torch_fp32",
                      "torch_fp16_amp"]
    assert [Variant.from_label(x) for x in labels] == plan
    assert variants(["cpu"], [False], ["fp32"]) == [Variant("ort", device="cpu")]
    with pytest.raises(ValueError, match="unknown variant"):
        Variant.from_label("tensorrt")


def test_speedup_is_baseline_over_other() -> None:
    rows = [{"variant": "a", "latency": {"p50_ms": 10.0}},
            {"variant": "b", "latency": {"p50_ms": 4.0}}]
    assert speedup(rows, "a", "b") == pytest.approx(2.5)
