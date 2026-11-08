"""Request / response models of the API (Phase 6a)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Defect(BaseModel):
    """One detection (steel) or one anomalous region (VisA), in original-image pixels."""

    label: str = Field(description="NEU-DET class name, or 'anomaly' for a region of the map")
    score: float
    box: tuple[float, float, float, float] = Field(description="x1, y1, x2, y2")


class OperatingPoint(BaseModel):
    fn_fp_ratio: float = Field(description="assumed cost of a miss, in false alarms")
    threshold: float = Field(description="frozen in Phase 4a on the selection split")
    frozen_file_sha256: str = Field(description="SHA-256 of reports/frozen_thresholds.json")


class AnomalyResult(BaseModel):
    score: float
    threshold: float
    flagged: bool
    map_size: tuple[int, int] = Field(description="(height, width) of the model's anomaly map")
    heatmap_png_base64: str | None = Field(
        None,
        description="overlay of the map on the image; the colour scale is map / (2 * threshold), "
        "so the threshold is mid-scale. Null when heatmap=false.",
    )


class Timing(BaseModel):
    decode_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    load_ms: float = Field(description="time to load the model for this call (0 when resident)")
    total_ms: float


class InspectResponse(BaseModel):
    product: str
    model: str
    image_size: tuple[int, int] = Field(description="(height, width) of the uploaded image")
    flagged: bool = Field(
        description="steel: at least one detection at or above the threshold; "
        "VisA: image score at or above the threshold"
    )
    defects: list[Defect]
    anomaly: AnomalyResult | None = None
    operating_point: OperatingPoint
    provider: str = Field(description="execution provider that ran this call")
    tf32: bool
    warnings: list[str] = []
    timing: Timing


class ProductInfo(BaseModel):
    name: str
    kind: Literal["detector", "anomaly"]
    model: str
    description: str
    available: bool
    reason: str = Field("", description="why it is unavailable, e.g. the ONNX file is missing")
    loaded: bool
    tf32: bool
    vram_mib: int
    thresholds: dict[str, float] = Field(description="frozen threshold per fn_fp_ratio")
    default_fn_fp_ratio: float


class LoadedInfo(BaseModel):
    product: str
    provider: str
    tf32: bool
    vram_mib: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    provider: str = Field(description="provider new sessions are created with")
    requested_providers: list[str]
    fallback_reason: str = ""
    onnxruntime: str
    frozen_file_sha256: str
    vram_budget_mib: int
    vram_in_use_mib: int = Field(description="sum of the declared sizes of the loaded sessions")
    loaded: list[LoadedInfo]
    available_products: list[str]
    unavailable_products: dict[str, str]
