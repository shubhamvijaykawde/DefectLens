"""Products -> ONNX Runtime sessions, loaded lazily under a device-memory budget (Phase 6a).

* A product is a name in ``configs/serve/default.yaml`` (``steel``, ``visa_pcb1``, ...). It maps to
  exactly one model, so steel and VisA are never fused.
* Sessions are created on first use, not at start-up. Each product declares the device memory of
  one session (the Phase 5a benchmark row); when loading one more would exceed
  ``vram_budget_mib`` the least recently used sessions are dropped first. The declared sizes
  include a CUDA context each, so the sum over-counts a shared context: a conservative estimate.
* Providers: ``providers`` in order of preference, filtered by what ONNX Runtime has. CUDA is the
  default and CPU the fallback; ``/health`` says which one is active and why.
* Session options and provider options are the ones Phase 5a verified (``export/verify.py``
  ``make_session``): default graph optimisation, ``use_tf32`` set explicitly per product.
* Thresholds come from ``reports/frozen_thresholds.json``, whose embedded hash is verified at
  start-up (``read_frozen``). An edited or missing file stops the API from starting.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import onnxruntime as ort

from defectlens.eval.thresholds import FrozenEntry, read_frozen
from defectlens.serve.config import ProductSpec, ServeSettings

log = logging.getLogger("defectlens.serve")

CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"
DOMAIN = {"detector": "neu_det", "anomaly": "visa"}


class UnknownProduct(KeyError):
    """No such product (the API answers 404)."""


class UnknownRatio(ValueError):
    """No frozen threshold for this fn_fp_ratio (the API answers 422)."""


class ModelUnavailable(RuntimeError):
    """The product exists but its model can not be served (the API answers 503)."""


@dataclass
class LoadedModel:
    product: str
    session: Any  # ort.InferenceSession
    input_name: str
    input_hw: tuple[int, int]
    provider: str
    tf32: bool
    vram_mib: int
    loaded_at: float = field(default_factory=time.time)


SessionFactory = Callable[[str, bool, list[str], bool], tuple[Any, str]]


# A 69-byte ONNX model (one Identity node), used to find out whether CUDA really works.
PROBE_MODEL = (
    b'\x08\x08:;\n\x10\n\x01X\x12\x01Y"\x08Identity\x12\x05probeZ\x0f\n\x01X\x12\n\n\x08\x08\x01'
    b'\x12\x04\n\x02\x08\x01b\x0f\n\x01Y\x12\n\n\x08\x08\x01\x12\x04\n\x02\x08\x01B\x04\n\x00\x10\r'
)


def probe_cuda() -> tuple[bool, str]:
    """Whether a CUDA session can be created in this process; ``(ok, why not)``.

    ``get_available_providers`` only says the CUDA library is part of the build. Without torch
    imported first (the slim image has none) ORT can not find the CUDA / cuDNN libraries that the
    ``onnxruntime-gpu[cuda]`` wheels install in site-packages, so they are preloaded here, and a
    missing library makes ORT drop the provider silently, which the probe turns into a reason.
    """
    try:
        ort.preload_dlls()
    except Exception as exc:  # older ORT, or libraries really absent
        log.warning("onnxruntime.preload_dlls failed: %s: %s", type(exc).__name__, exc)
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(PROBE_MODEL, so, providers=[CUDA, CPU])
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
    if sess.get_providers()[0] == CUDA:
        return True, ""
    return False, (
        f"{CUDA} could not be loaded (usually a missing CUDA 13 / cuDNN 9 library; "
        "see the onnxruntime warning in the log)"
    )


def create_session(
    path: str, tf32: bool, providers: list[str], require_cuda: bool
) -> tuple[Any, str]:
    """An ORT session on the first usable provider; returns it with the provider it runs on."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if CUDA in providers:
        try:
            sess = ort.InferenceSession(
                path, so,
                providers=[(CUDA, {"use_tf32": "1" if tf32 else "0"}), CPU],
            )
            if sess.get_providers()[0] == CUDA:
                return sess, CUDA
            reason = f"CUDA did not load for {path}: {sess.get_providers()}"
        except Exception as exc:  # no GPU / driver / out of memory
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
        if require_cuda:
            msg = f"require_cuda is set and {reason}"
            raise ModelUnavailable(msg)
        log.warning("falling back to CPU for %s (%s)", path, reason)
    return ort.InferenceSession(path, so, providers=[CPU]), CPU


def _input_hw(sess: Any) -> tuple[int, int]:
    shape = sess.get_inputs()[0].shape
    h, w = shape[-2], shape[-1]
    if not (isinstance(h, int) and isinstance(w, int)):
        msg = f"the model input has a symbolic spatial shape {shape}; expected fixed H and W"
        raise ModelUnavailable(msg)
    return h, w


class ModelRegistry:
    def __init__(
        self, settings: ServeSettings, *, session_factory: SessionFactory = create_session,
        available_providers: list[str] | None = None,
    ) -> None:
        self.settings = settings
        self._factory = session_factory
        avail = (
            ort.get_available_providers() if available_providers is None else available_providers
        )
        self.providers = [p for p in settings.providers if p in avail] or [CPU]
        wanted_cuda = CUDA in settings.providers
        self.fallback_reason = (
            f"{CUDA} is not available in this onnxruntime build ({avail})"
            if wanted_cuda and CUDA not in avail else ""
        )
        if CUDA in self.providers:
            ok, why = probe_cuda()
            if not ok:
                self.providers = [p for p in self.providers if p != CUDA] or [CPU]
                self.fallback_reason = why
                log.warning("CUDA is not usable, serving on CPU: %s", why)
        if settings.require_cuda and CUDA not in self.providers:
            msg = f"require_cuda is set but {self.fallback_reason or 'CUDA was not requested'}"
            raise ModelUnavailable(msg)
        entries, meta = read_frozen(settings.frozen_thresholds)
        self._entries: dict[tuple[str, str, str], list[FrozenEntry]] = {}
        for e in entries:
            self._entries.setdefault(e.key, []).append(e)
        self.frozen_meta = meta
        self.frozen_sha256 = hashlib.sha256(settings.frozen_thresholds.read_bytes()).hexdigest()
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.RLock()

    # ---- products -------------------------------------------------------------------------

    def spec(self, product: str) -> ProductSpec:
        try:
            return self.settings.products[product]
        except KeyError:
            raise UnknownProduct(product) from None

    def _entries_of(self, spec: ProductSpec) -> list[FrozenEntry]:
        return self._entries.get(
            (DOMAIN[spec.kind], spec.frozen_model, spec.frozen_category), []
        )

    def ratios(self, product: str) -> list[float]:
        return sorted(e.ratio for e in self._entries_of(self.spec(product)))

    def threshold(self, product: str, ratio: float) -> FrozenEntry:
        spec = self.spec(product)
        for e in self._entries_of(spec):
            if math.isclose(e.ratio, ratio, rel_tol=0, abs_tol=1e-9):
                return e
        msg = (
            f"no frozen threshold for fn_fp_ratio={ratio:g} for {product!r}; "
            f"available: {[f'{r:g}' for r in self.ratios(product)]}"
        )
        raise UnknownRatio(msg)

    def unavailable_reason(self, product: str) -> str:
        """'' if the product can be served, else why not (no session is created to find out)."""
        spec = self.spec(product)
        if not spec.onnx.is_file():
            return f"model file {spec.onnx} not found"
        if not self._entries_of(spec):
            return (
                f"no frozen threshold for {spec.frozen_model}/{spec.frozen_category} in "
                f"{self.settings.frozen_thresholds}"
            )
        return ""

    # ---- sessions -------------------------------------------------------------------------

    def acquire(self, product: str) -> LoadedModel:
        """The loaded model of ``product``, loading it (and evicting others) if needed."""
        spec = self.spec(product)
        with self._lock:
            if product in self._loaded:
                self._loaded.move_to_end(product)
                return self._loaded[product]
            reason = self.unavailable_reason(product)
            if reason:
                raise ModelUnavailable(reason)
            budget = self.settings.vram_budget_mib
            while self._loaded and self.vram_in_use() + spec.vram_mib > budget:
                old, _ = self._loaded.popitem(last=False)
                log.info("evicting %s to make room for %s", old, product)
            try:
                sess, provider = self._factory(
                    str(spec.onnx), spec.tf32, self.providers, self.settings.require_cuda
                )
                model = LoadedModel(
                    product, sess, sess.get_inputs()[0].name, _input_hw(sess), provider,
                    spec.tf32, spec.vram_mib,
                )
            except ModelUnavailable:
                raise
            except Exception as exc:
                msg = f"could not load {spec.onnx}: {type(exc).__name__}"
                log.exception(msg)
                raise ModelUnavailable(msg) from exc
            self._loaded[product] = model
            log.info("loaded %s on %s (tf32=%s)", product, provider, spec.tf32)
            return model

    def vram_in_use(self) -> int:
        return sum(m.vram_mib for m in self._loaded.values())

    def loaded(self) -> list[LoadedModel]:
        with self._lock:
            return list(self._loaded.values())

    def evict_all(self) -> None:
        with self._lock:
            self._loaded.clear()
