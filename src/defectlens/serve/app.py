"""FastAPI app (Phase 6a): ``POST /inspect``, ``GET /health``, ``GET /products``.

``create_app`` builds the app from settings (tests pass their own); ``uvicorn ... --factory``
calls it with none and reads ``$DEFECTLENS_SERVE_CONFIG`` / ``configs/serve/default.yaml``.

Errors: 400 not an image, 404 unknown product, 413 upload too large, 422 bad parameter (the
framework's own for a missing field), 503 the product's model can not be served (file missing,
failed to load), 500 anything else (logged, details not returned).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from defectlens.serve.config import ServeSettings, load_settings
from defectlens.serve.inference import BadImage, Inspector
from defectlens.serve.model_registry import (
    ModelRegistry,
    ModelUnavailable,
    UnknownProduct,
    UnknownRatio,
)
from defectlens.serve.schemas import (
    HealthResponse,
    InspectResponse,
    LoadedInfo,
    ProductInfo,
)

log = logging.getLogger("defectlens.serve")


def create_app(
    settings: ServeSettings | None = None, *, registry: ModelRegistry | None = None
) -> FastAPI:
    cfg = settings or load_settings()
    reg = registry or ModelRegistry(cfg)
    inspector = Inspector(cfg, reg)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        for product in cfg.preload:
            try:
                reg.acquire(product)
            except ModelUnavailable as exc:
                log.warning("preload of %s failed: %s", product, exc)
        yield
        reg.evict_all()

    app = FastAPI(
        title="DefectLens", version="0.1.0", lifespan=lifespan,
        description="Industrial visual inspection: steel defects (NEU-DET detector) and "
        "VisA anomaly products, served with ONNX Runtime.",
    )
    app.state.registry = reg
    app.state.inspector = inspector

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        names = list(cfg.products)
        reasons = {n: r for n in names if (r := reg.unavailable_reason(n))}
        available = [n for n in names if n not in reasons]
        return HealthResponse(
            status="ok" if available else "degraded",
            provider=reg.providers[0],
            requested_providers=cfg.providers,
            fallback_reason=reg.fallback_reason,
            onnxruntime=ort.__version__,
            frozen_file_sha256=reg.frozen_sha256,
            vram_budget_mib=cfg.vram_budget_mib,
            vram_in_use_mib=reg.vram_in_use(),
            loaded=[
                LoadedInfo(product=m.product, provider=m.provider, tf32=m.tf32, vram_mib=m.vram_mib)
                for m in reg.loaded()
            ],
            available_products=available,
            unavailable_products=reasons,
        )

    @app.get("/products", response_model=list[ProductInfo])
    def products() -> list[ProductInfo]:
        loaded = {m.product for m in reg.loaded()}
        out = []
        for name, spec in cfg.products.items():
            reason = reg.unavailable_reason(name)
            out.append(ProductInfo(
                name=name, kind=spec.kind, model=spec.frozen_model, description=spec.description,
                available=not reason, reason=reason, loaded=name in loaded, tf32=spec.tf32,
                vram_mib=spec.vram_mib,
                thresholds={f"{r:g}": reg.threshold(name, r).threshold for r in reg.ratios(name)},
                default_fn_fp_ratio=cfg.fn_fp_ratio,
            ))
        return out

    @app.post("/inspect", response_model=InspectResponse)
    def inspect(
        image: Annotated[UploadFile, File(description="JPEG, PNG or BMP")],
        product: Annotated[str, Query(description="steel, visa_cashew, visa_candle, ...")],
        fn_fp_ratio: Annotated[
            float | None,
            Query(description="frozen operating point; default from the settings (5, 10 or 50)"),
        ] = None,
        heatmap: Annotated[bool, Query(description="anomaly products: include the overlay")] = True,
    ) -> InspectResponse:
        data = image.file.read(cfg.max_upload_bytes + 1)
        if len(data) > cfg.max_upload_bytes:
            raise HTTPException(413, f"upload is larger than {cfg.max_upload_bytes} bytes")
        try:
            return inspector.inspect(product, data, fn_fp_ratio=fn_fp_ratio, heatmap=heatmap)
        except UnknownProduct:
            raise HTTPException(
                404, f"unknown product {product!r}; available: {sorted(cfg.products)}"
            ) from None
        except UnknownRatio as exc:
            raise HTTPException(422, str(exc)) from None
        except BadImage as exc:
            raise HTTPException(400, str(exc)) from None
        except ModelUnavailable as exc:
            raise HTTPException(503, f"product {product!r} is not available: {exc}") from None
        except Exception:
            log.exception("inspect failed for product %s", product)
            raise HTTPException(500, "internal error while inspecting the image") from None

    return app
