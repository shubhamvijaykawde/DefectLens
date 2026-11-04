# Phase 0c spike notes

Run 2026-11-04 in WSL2 Ubuntu on the RTX 4060 Laptop (8 GB). Every number below was printed by the
scripts named next to it. Nothing here is a reported result.

## Versions (from `uv.lock`)

torch 2.14.1+cu130, torchvision 0.29.1+cu130, transformers 5.22.1, anomalib 2.7.0, timm 1.0.31,
onnx 1.23.2, onnxscript 0.7.2, onnxruntime-gpu 1.31.1, mlflow 3.18.0, dvc 3.67.1.

## D-FINE-S (`scripts/spike_dfine.py`)

- Checkpoint `ustc-community/dfine-small-coco` (Hub card lists apache-2.0; weights terms still to be
  checked per `PLAN.md` section 3). 10.3 M parameters.
- Forward pass, batch 1, 640x640, fp16 autocast: logits `(1, 300, 80)`, boxes `(1, 300, 4)`.
- Peak VRAM, inference: **124 MiB**.

## ONNX export and ORT CUDA (`scripts/spike_export.py`): required gate PASSED, with three findings

- Dynamo path (`torch.onnx.export(dynamo=True)`, needs `onnxscript`) wrote a 51.6 MiB file, **opset 20**.
- Gate result: ORT `CUDAExecutionProvider`, fp32, TF32 off, max abs diff vs PyTorch
  **7.0e-05 (logits), 9.6e-06 (boxes)**, pass threshold 1e-3. This is on a random input, so it proves
  numerical agreement of the graph, not detection quality.

Findings that Phase 5a must carry forward:

1. **Export needs `TRANSFORMERS_DISABLE_TORCH_CHECK=1`.** Without it the translator fails with
   `No ONNX function found for aten._is_all_true`, from a shape sanity check in D-FINE's deformable
   attention (`torch_compilable_check`). The variable is documented by transformers for export
   and production. The input size is fixed at 640, so the check is not protecting anything here.
2. **Export with `optimize=False`.** With the exporter's own optimization on, ORT 1.31.1's default
   graph optimizer returned wrong outputs for the HGNetV2 stem (max diff 1.47 on the embedder alone).
   The same graph is correct if ORT optimization is disabled, and an unoptimized export is correct
   under ORT's default optimization. Only the combination fails. Isolated by bisecting
   stage by stage; this points at an ORT optimizer fusion, but I did not identify the exact pass.
3. **ORT's default TF32 on CUDA changes D-FINE outputs completely.** With `use_tf32=1` (the default)
   the same model differs from PyTorch by 3.6 (logits) and 0.98 (boxes). With `use_tf32=0` it matches
   to 7e-05. Mechanism, as far as tested: TF32 shifts early convolution outputs by about 7e-3
   (measured on the stem), and top-k proposal selection amplifies it into different queries. Whether
   this costs any **mAP** is not measured. Phase 5a must compare detection metrics with TF32 on and
   off before choosing a default, and benchmark latency both ways.

TensorRT probe (informational): `TensorrtExecutionProvider` appears in `ort.get_available_providers()`
for 1.31.1. I did not create a TensorRT session, so whether it actually works is untested (Phase 5b).

## AnomalyDINO (`scripts/spike_anomalydino.py category=cashew shots=5`)

Encoder, settled in the plan's order, before any data was used:

1. `anomalib.__version__` is 2.7.0; source at `.venv/.../anomalib/models/image/anomaly_dino/`.
2. `encoder_name` default is `vit_small_patch14_dinov2` in both `lightning_model.py` and
   `torch_model.py` (signature and source text agree).
3. timm resolved the name and instantiated the model.
4. Forward pass on a dummy `(2, 3, 252, 252)` tensor: features `(2, 324, 384)`, patch size 14.

Pinned in `configs/model/anomaly_dino.yaml`.

End to end on VisA `cashew`, 5-shot, seed 42, input 252: image AUROC **0.9304**, pixel AUROC 0.9900.
**Smoke values only.** They use the official test set, one draw, and no thresholds.
Peak VRAM (torch allocator) **586 MiB**, wall time about 21 s with the weights cached.

Correction recorded: my first version of the spike set the training subset after `setup()`, and
`Engine.fit` rebuilt the dataset, so it silently fitted all 450 images (memory bank 145,800 vectors,
AUROC 0.9654, 4,685 MiB). The script now subsets inside `_setup` and asserts the memory bank is
`shots x 324`. Phase 3c's few-shot code must keep that assertion.

## VisA download and versioning

- 1.93 GB tar from the official AWS URL, SHA-256 checked against the hash shipped in anomalib 2.7.0
  (matched). The first attempt died at 1.58 GB on a broken pipe; the downloader now retries and resumes.
- Extracted to `data/raw/visa` (24,058 files). `dvc add` recorded md5
  `e01ee2ec00343ffc082ea050a2e10556.dir`, 3,844,680,698 bytes, in `data/raw/visa.dvc`.
- The archive contains `LICENSE-DATASET`, whose text begins "Attribution 4.0 International"
  (CC BY 4.0). That is one more data point for `docs/DATA_TERMS.md` in Phase 1a; the conflict with other
  sources listed in `PLAN.md` is not resolved by this.

## Plan estimates vs measured

| Item | Estimate | Measured |
|---|---|---|
| D-FINE spike VRAM | 1-2 GB | 124 MiB (inference only; training will be higher) |
| AnomalyDINO 5-shot VRAM | 1-2 GB | 586 MiB |
| AnomalyDINO full-train-set fit (accident) | n/a | 4,685 MiB |
| VisA download | 10-30 min, about 2 GB | 1.93 GB; over an hour at about 0.5-1 MB/s with one interruption |
