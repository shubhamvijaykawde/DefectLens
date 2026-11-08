"""Latency benchmark of Phase 5a: batch 1, warm-up, many timed runs, p50 / p95, throughput.

Protocol (fixed in ``configs/export/default.yaml``): batch 1, 50 warm-up calls, 500 timed calls of
one fixed real image (a selection-split image, so no evaluation image is read). The timed call is
the model only, from a numpy / CPU tensor input to the outputs back on the host: for ONNX Runtime
that is ``session.run`` (host-to-device copy, kernels, device-to-host copy), for PyTorch it is
the forward pass followed by ``.cpu()`` (which synchronises). Pre- and post-processing (letterbox,
top-k decode, heatmap) are **not** included, so these are model latencies, not API latencies.

Each (model, variant) is measured in its own process (``scripts/benchmark.py`` starts it), so the
peak RAM and the device memory belong to that variant alone. Device memory is the device-wide
``memory.used`` sampled while the variant runs, minus the value read before this process touched
the GPU; it includes the CUDA context. Other programs using the GPU (Windows apps under WSL) add
noise to it. Numbers come from WSL2.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


def summarize(times_ms: Sequence[float]) -> dict[str, float]:
    """Latency summary of a list of per-call times in milliseconds.

    ``throughput_per_s`` is sequential batch-1 throughput, ``1000 / mean``.
    """
    if not times_ms:
        msg = "no timings to summarise"
        raise ValueError(msg)
    t = np.asarray(times_ms, dtype=np.float64)
    return {
        "n": float(len(t)), "mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
        "p95_ms": float(np.percentile(t, 95)), "min_ms": float(t.min()), "max_ms": float(t.max()),
        "std_ms": float(t.std()), "throughput_per_s": float(1000.0 / t.mean()),
    }


def time_calls(call: Callable[[], object], *, warmup: int, runs: int) -> list[float]:
    """``warmup`` untimed calls, then ``runs`` timed ones; milliseconds per call."""
    if warmup < 0 or runs < 1:
        msg = f"need warmup >= 0 and runs >= 1, got {warmup}, {runs}"
        raise ValueError(msg)
    for _ in range(warmup):
        call()
    out: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        call()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def nvidia_smi_used_mib() -> float | None:
    """Device-wide used memory in MiB from ``nvidia-smi`` (no CUDA context needed), or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip().splitlines()
        return float(out[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


@dataclass
class MemorySampler:
    """Background thread keeping the maximum of ``read()`` (MiB) while the block runs."""

    read: Callable[[], float]
    period_s: float = 0.02
    peak: float = field(default=0.0, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.read())
            self._stop.wait(self.period_s)

    def __enter__(self) -> MemorySampler:
        self.peak = self.read()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak = max(self.peak, self.read())


@dataclass(frozen=True)
class Variant:
    """One way to run one model: ``kind`` "ort" (with ``tf32``) or "torch" (with ``precision``)."""

    kind: str
    tf32: bool = False
    precision: str = "fp32"
    device: str = "cuda"

    @property
    def label(self) -> str:
        if self.kind == "ort":
            if self.device == "cpu":
                return "ort_cpu"
            return f"ort_cuda_tf32_{'on' if self.tf32 else 'off'}"
        return f"torch_{self.precision}"

    @classmethod
    def from_label(cls, label: str) -> Variant:
        if label == "ort_cpu":
            return cls("ort", device="cpu")
        if label.startswith("ort_cuda_tf32_"):
            return cls("ort", tf32=label.endswith("_on"))
        if label.startswith("torch_"):
            return cls("torch", precision=label.removeprefix("torch_"))
        msg = f"unknown variant {label!r}"
        raise ValueError(msg)


def variants(
    providers: Sequence[str], tf32: Sequence[bool], torch_variants: Sequence[str]
) -> list[Variant]:
    """ORT variants for each provider (CUDA: one per TF32 setting) and the PyTorch references."""
    out: list[Variant] = []
    if "cuda" in providers:
        out += [Variant("ort", tf32=bool(t)) for t in tf32]
    if "cpu" in providers:
        out.append(Variant("ort", device="cpu"))
    if "cuda" in providers:
        out += [Variant("torch", precision=p) for p in torch_variants]
    return out


def speedup(
    rows: Sequence[dict[str, Any]], baseline: str, other: str, key: str = "p50_ms"
) -> float:
    """``baseline`` latency over ``other`` latency for one model (>1: ``other`` is faster)."""
    get = {r["variant"]: r["latency"][key] for r in rows}
    return float(get[baseline] / get[other])


def as_array(x: Any) -> npt.NDArray[np.float32]:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))
