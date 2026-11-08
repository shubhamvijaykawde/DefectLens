"""Settings of the API (Phase 6a): ``configs/serve/default.yaml`` validated with pydantic.

numpy / pydantic / PyYAML only: the serving image has no torch (Phase 6b), so nothing under
``defectlens.serve`` may import it (``tests/test_api.py`` checks this in a subprocess).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG = Path("configs/serve/default.yaml")
CONFIG_ENV = "DEFECTLENS_SERVE_CONFIG"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HeatmapSettings(Strict):
    max_side: int = Field(512, gt=0, description="longest side of the returned overlay, px")
    alpha: float = Field(0.45, ge=0.0, le=1.0, description="weight of the colour map")
    max_regions: int = Field(10, gt=0, description="most regions listed per anomaly image")


class DFineSettings(Strict):
    top_k: int = Field(100, gt=0)
    score_floor: float = Field(0.001, ge=0.0)


class ProductSpec(Strict):
    """One ``product``: the model that serves it and where its frozen threshold lives."""

    kind: Literal["detector", "anomaly"]
    # ONNX output contract (see export/onnx_export.py): "dfine" = logits + normalised cxcywh
    # boxes, "torchvision" = boxes / scores / labels, "anomaly" = score + anomaly map.
    format: Literal["dfine", "torchvision", "anomaly"]
    onnx: Path
    frozen_model: str = Field(description="model name in frozen_thresholds.json")
    frozen_category: str = Field(description="'all' for NEU-DET, the VisA category otherwise")
    tf32: bool = False
    vram_mib: int = Field(gt=0, description="device memory of one session (Phase 5a benchmark)")
    description: str = ""
    trained_image_size: tuple[int, int] | None = Field(
        None, description="(height, width) the model was trained on; other sizes get a warning"
    )

    @model_validator(mode="after")
    def _format_matches_kind(self) -> ProductSpec:
        if (self.kind == "anomaly") != (self.format == "anomaly"):
            msg = f"format {self.format!r} does not fit kind {self.kind!r}"
            raise ValueError(msg)
        return self


class ServeSettings(Strict):
    frozen_thresholds: Path
    fn_fp_ratio: float = Field(10.0, gt=0)
    providers: list[str] = Field(["CUDAExecutionProvider", "CPUExecutionProvider"])
    require_cuda: bool = False
    vram_budget_mib: int = Field(5000, gt=0)
    preload: list[str] = Field(default_factory=list)
    max_upload_bytes: int = Field(20 * 2**20, gt=0)
    max_pixels: int = Field(40_000_000, gt=0)
    heatmap: HeatmapSettings = Field(default_factory=lambda: HeatmapSettings.model_validate({}))
    dfine: DFineSettings = Field(default_factory=lambda: DFineSettings.model_validate({}))
    # Mirrors configs/preprocess/default.yaml (tests/test_serve_inference.py compares the two);
    # `size` is not here: it comes from the input shape of the ONNX file.
    steel_preprocess: dict[str, Any] = Field(default_factory=dict)
    products: dict[str, ProductSpec]

    @model_validator(mode="after")
    def _preload_is_known(self) -> ServeSettings:
        unknown = [p for p in self.preload if p not in self.products]
        if unknown:
            msg = f"preload names unknown products: {unknown}"
            raise ValueError(msg)
        return self


def load_settings(path: Path | str | None = None) -> ServeSettings:
    """Read the settings file: ``path``, else ``$DEFECTLENS_SERVE_CONFIG``, else the default."""
    p = Path(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG)
    doc = yaml.safe_load(p.read_text())
    return ServeSettings.model_validate(doc)
