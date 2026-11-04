"""Spike: Dynamo ONNX export of D-FINE-S, verify in ONNX Runtime CUDA vs PyTorch (fp32)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import DFineForObjectDetection

from defectlens.seed import seed_everything

# transformers documents this switch for production/export: it disables shape sanity checks
# (torch_compilable_check) that the Dynamo ONNX translator cannot convert (aten._is_all_true).
os.environ.setdefault("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")

MODEL_ID = "ustc-community/dfine-small-coco"
OUT = Path("models/spike/dfine_s.onnx")
ATOL = 1e-3


class Wrapper(torch.nn.Module):
    """Tensor-in, tuple-out wrapper so the exporter sees plain tensors."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


def main() -> int:
    seed_everything(42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = DFineForObjectDetection.from_pretrained(MODEL_ID).eval()
    wrapped = Wrapper(model).eval()
    x = torch.randn(1, 3, 640, 640)

    print("exporting with torch.onnx.export(dynamo=True) ...")
    # optimize=False: with the exporter's own graph optimization on, ORT 1.31.1's default graph
    # optimizer produces wrong outputs for the HGNetV2 stem (max diff ~1.5 on the stem alone,
    # fine when either optimizer is skipped). See reports/spike_notes.md.
    prog = torch.onnx.export(
        wrapped,
        (x,),
        dynamo=True,
        optimize=False,
        input_names=["pixel_values"],
        output_names=["logits", "boxes"],
    )
    assert prog is not None
    prog.save(str(OUT))
    m = onnx.load(str(OUT))
    onnx.checker.check_model(m)
    opsets = {o.domain or "ai.onnx": o.version for o in m.opset_import}
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.1f} MiB), opset {opsets}")

    with torch.no_grad():
        ref_logits, ref_boxes = (t.numpy() for t in wrapped(x))

    print(
        "TensorRT EP available (probe only): "
        f"{'TensorrtExecutionProvider' in ort.get_available_providers()}"
    )
    so = ort.SessionOptions()
    so.log_severity_level = 3
    results: dict[str, tuple[float, float]] = {}
    for tf32 in ("0", "1"):
        providers = [("CUDAExecutionProvider", {"use_tf32": tf32}), "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(OUT), so, providers=providers)
        active = sess.get_providers()
        got_logits, got_boxes = sess.run(None, {"pixel_values": x.numpy()})
        results[tf32] = (
            float(np.abs(got_logits - ref_logits).max()),
            float(np.abs(got_boxes - ref_boxes).max()),
        )
        print(
            f"ORT CUDA use_tf32={tf32}: providers {active}  "
            f"max abs diff logits {results[tf32][0]:.3e} boxes {results[tf32][1]:.3e}"
        )
    # TF32 (ORT's default on Ampere+) shifts early conv outputs by ~1e-2; D-FINE's top-k proposal
    # selection amplifies that into totally different queries, so element-wise comparison only
    # makes sense with TF32 off. Its effect on mAP must be measured later (Phase 5a), not assumed.
    ok = active[0] == "CUDAExecutionProvider" and max(results["0"]) < ATOL
    print(f"pass criterion: use_tf32=0 diff < {ATOL}")
    print("RESULT", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
