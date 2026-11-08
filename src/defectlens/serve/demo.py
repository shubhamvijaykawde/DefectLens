"""Gradio demo (Phase 6b), mounted on the API at ``/demo``.

The demo calls the same ``Inspector`` as ``POST /inspect`` in the same process, with the encoded
upload as bytes, so a demo answer is the API answer (same decoding, preprocessing, session,
frozen threshold), only rendered: steel gets its boxes drawn on the image, a VisA product gets the
heatmap overlay the API returns. Nothing is evaluated or tuned here.

``create_app_with_demo`` is the uvicorn factory of the Docker image::

    uvicorn --factory defectlens.serve.demo:create_app_with_demo --host 0.0.0.0

The product list is read when the app is built: a product whose ONNX file is missing at that time
is not offered (the API's ``/products`` says why).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import numpy.typing as npt
from fastapi import FastAPI

from defectlens.serve.app import create_app
from defectlens.serve.config import ServeSettings
from defectlens.serve.inference import BadImage, Inspector, decode_image
from defectlens.serve.model_registry import ModelUnavailable, UnknownProduct, UnknownRatio
from defectlens.serve.schemas import Defect, InspectResponse

log = logging.getLogger("defectlens.serve")

DEMO_PATH = "/demo"
BOX_COLORS = [  # RGB; a class gets the colour of its rank among the labels in the picture
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
]
NOTE = (
    "Scores are model scores, not probabilities. The thresholds were frozen in Phase 4a on a "
    "selection split for *assumed* miss : false-alarm cost ratios; the ONNX path moves some "
    "scores slightly (`reports/serve_models.md`). A research demo, not a safety system."
)


class DemoError(gr.Error):
    """An error the user should see in the UI (bad image, unknown product, model missing)."""


def draw_defects(bgr: npt.NDArray[np.uint8], defects: list[Defect]) -> npt.NDArray[np.uint8]:
    """RGB copy of ``bgr`` with each defect's box and ``label score`` drawn on it."""
    out = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    labels = sorted({d.label for d in defects})
    thick = max(1, round(max(out.shape[:2]) / 250))
    scale = max(0.4, max(out.shape[:2]) / 700)
    for d in defects:
        colour = BOX_COLORS[labels.index(d.label) % len(BOX_COLORS)]
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick)
        text = f"{d.label} {d.score:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        top = max(0, y1 - th - base)
        cv2.rectangle(out, (x1, top), (x1 + tw, top + th + base), colour, -1)
        cv2.putText(out, text, (x1, top + th), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    thick, cv2.LINE_AA)
    return out


def decode_overlay(png_b64: str) -> npt.NDArray[np.uint8]:
    """The API's base64 PNG heatmap overlay as an RGB array."""
    buf = np.frombuffer(base64.b64decode(png_b64), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        msg = "the heatmap is not a decodable PNG"
        raise ValueError(msg)
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def summarize(resp: InspectResponse) -> str:
    """Markdown summary of one answer: verdict, operating point, execution details, warnings."""
    op = resp.operating_point
    if resp.anomaly is not None:
        a = resp.anomaly
        verdict = "ANOMALY" if resp.flagged else "no anomaly flagged"
        head = (
            f"### {resp.product}: {verdict}\n\nimage score **{a.score:.4g}** against threshold "
            f"**{a.threshold:.4g}** (flagged when score >= threshold)."
        )
    else:
        n = len(resp.defects)
        verdict = f"{n} defect{'s' if n != 1 else ''}" if resp.flagged else "no defect"
        head = (
            f"### {resp.product}: {verdict}\n\n"
            f"boxes with score >= **{op.threshold:.4g}** are kept."
        )
    t = resp.timing
    lines = [
        head,
        f"- operating point: miss : false-alarm cost ratio {op.fn_fp_ratio:g}, frozen Phase 4a "
        f"threshold (file sha256 `{op.frozen_file_sha256[:12]}`)",
        f"- model `{resp.model}` on `{resp.provider}`, TF32 {'on' if resp.tf32 else 'off'}, "
        f"{resp.image_size[1]}x{resp.image_size[0]} px input",
        f"- {t.total_ms:.0f} ms in the server (model {t.inference_ms:.0f} ms, model load "
        f"{t.load_ms:.0f} ms), network not included",
    ]
    lines += [f"- warning: {w}" for w in resp.warnings]
    return "\n".join(lines)


def inspect_for_demo(
    inspector: Inspector, product: str | None, image_path: str | None, ratio: str | float | None,
    heatmap: bool = True,
) -> tuple[npt.NDArray[np.uint8], str, list[list[Any]], dict[str, Any]]:
    """What the Gradio button runs: ``(picture, markdown, defect rows, full API answer)``.

    Raises ``DemoError`` with a readable message instead of the API's HTTP errors.
    """
    if not product:
        raise DemoError("Pick a product first.")
    if not image_path:
        raise DemoError("Upload an image first.")
    data = Path(image_path).read_bytes()
    try:
        resp = inspector.inspect(
            product, data, fn_fp_ratio=None if ratio in (None, "") else float(ratio),
            heatmap=heatmap,
        )
        image = decode_image(data, color=True, max_pixels=inspector.settings.max_pixels)
    except UnknownProduct:
        raise DemoError(f"Unknown product {product!r}.") from None
    except UnknownRatio as exc:
        raise DemoError(str(exc)) from None
    except BadImage as exc:
        raise DemoError(f"Not a usable image: {exc}") from None
    except ModelUnavailable as exc:
        raise DemoError(f"The model for {product!r} is not available: {exc}") from None
    if resp.anomaly is not None and resp.anomaly.heatmap_png_base64:
        picture = decode_overlay(resp.anomaly.heatmap_png_base64)
    else:
        picture = draw_defects(image, resp.defects)
    rows = [[d.label, round(d.score, 4), *(round(v, 1) for v in d.box)] for d in resp.defects]
    shown = resp.model_dump(exclude={"anomaly": {"heatmap_png_base64"}})
    return picture, summarize(resp) + "\n\n" + NOTE, rows, shown


def build_demo(inspector: Inspector) -> gr.Blocks:
    """The UI: product, image, operating point, heatmap switch; picture, summary, table, JSON."""
    reg, cfg = inspector.registry, inspector.settings
    products = [p for p in cfg.products if not reg.unavailable_reason(p)]
    ratios = sorted({r for p in cfg.products for r in reg.ratios(p)})
    ratio_choices = [f"{r:g}" for r in ratios]
    default_ratio = f"{cfg.fn_fp_ratio:g}"
    describe = {p: cfg.products[p].description for p in cfg.products}

    with gr.Blocks(title="DefectLens demo") as demo:
        gr.Markdown(
            "# DefectLens\nSteel surface defects (NEU-DET detector) and VisA anomaly products, "
            "served with ONNX Runtime. Upload an image of the product you pick."
        )
        with gr.Row():
            with gr.Column():
                product = gr.Dropdown(
                    products, value=products[0] if products else None, label="Product",
                    info=describe[products[0]] if products else "no model file is available",
                )
                image = gr.Image(type="filepath", label="Image (JPEG, PNG or BMP)", height=360)
                ratio = gr.Radio(
                    ratio_choices, value=default_ratio if default_ratio in ratio_choices else None,
                    label="Assumed cost of a miss, in false alarms",
                    info="Selects the frozen threshold; an assumption, not a measurement.",
                )
                heat = gr.Checkbox(True, label="Heatmap overlay (anomaly products)")
                run = gr.Button("Inspect", variant="primary")
            with gr.Column():
                picture = gr.Image(label="Result", interactive=False, height=360)
                summary = gr.Markdown()
        table = gr.Dataframe(
            headers=["label", "score", "x1", "y1", "x2", "y2"], interactive=False,
            label="Defects (steel) or anomalous regions (VisA), original-image pixels",
        )
        with gr.Accordion("Full API answer", open=False):
            raw = gr.JSON()
        product.change(lambda p: gr.Dropdown(info=describe.get(p, "")), product, product)
        run.click(
            lambda p, i, r, h: inspect_for_demo(inspector, p, i, r, h),
            [product, image, ratio, heat], [picture, summary, table, raw],
            api_name="inspect",
        )
    demo_blocks: gr.Blocks = demo
    return demo_blocks


def mount_demo(app: FastAPI, path: str = DEMO_PATH) -> FastAPI:
    """Add the demo to an app made by ``create_app`` (it reads ``app.state.inspector``)."""
    demo = build_demo(app.state.inspector)
    mounted: FastAPI = gr.mount_gradio_app(app, demo, path=path)
    return mounted


def create_app_with_demo(settings: ServeSettings | None = None) -> FastAPI:
    """The API plus the demo at ``/demo`` (uvicorn factory)."""
    app = mount_demo(create_app(settings))
    log.info("demo at %s", DEMO_PATH)
    return app
