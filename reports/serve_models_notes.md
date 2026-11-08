Hand-written after reading the tables above. Numbers are from the tables above, from `reports/anomaly_comparison.md`,
`reports/thresholds.md`, `reports/benchmark.md`, or from the live run described under "Live check". One machine, one run per
measurement, one seed. Nothing here read an evaluation split (`threshold_evaluation`, `test`); the flip counts use the
selection splits only.

## 1. Which models the API serves

One model per product, loaded lazily under a device-memory budget of 5,000 MiB (the declared sizes of all five products add up
to 4,218 MiB, so they fit together; the LRU eviction only matters if a larger model is added).

| product | model | why |
|---|---|---|
| `steel` | Faster R-CNN (the augmented run) | lowest cost on `val` at every ratio, under PyTorch and under ORT (below) |
| `visa_pcb1`, `visa_macaroni2`, `visa_cashew` | EfficientAD-S | the cheapest model, exported for all four categories (31 MiB, 574-642 MiB on the device); leads on pcb1 and macaroni2, and is not clearly beaten on cashew |
| `visa_candle` | PatchCore | the only category where a heavier model leads EfficientAD by more than 0.05 image AUROC on `threshold_calibration` |

**Anomaly rule (mine, written down before the flips were counted):** EfficientAD unless another model beats it by more than 0.05
image AUROC on `threshold_calibration`, the selection split (`reports/anomaly_comparison.md`, full data). That table gives cashew
PatchCore 0.9464 / AnomalyDINO 0.9624 / EfficientAD 0.9304 (best minus EfficientAD: +0.032, not enough), candle 0.9736 / 0.9572 / 0.9108
(+0.063, PatchCore), pcb1 0.9188 / 0.9304 / 0.9716 and macaroni2 0.7208 / 0.7336 / 0.8268 (EfficientAD best in both). The
0.05 cut-off is a judgement: I had read the AUROC table when I chose it, and with 75-100 images and one seed a gap of 0.03 and a
gap of 0.06 are both small. A reader who prefers the highest AUROC per category gets AnomalyDINO for cashew and the same
choices elsewhere.

- **AnomalyDINO is not served.** Under ORT it needs 4,942-4,960 MiB on the device (`reports/benchmark.md`) and its ONNX file is
  515 MiB per category; no category passes the 0.05 rule. It is not in the config; adding it would evict every other session.
- **PatchCore candle** needed a new ONNX export (`models/onnx/patchcore_candle.onnx`, 137.6 MiB, 4.3 s on CPU, same exporter
  and opset as the verified pcb1 export). It was not put through the Phase 5a verification (AUROC / Spearman), only through the
  flip count here: max relative score difference to PyTorch 3.35e-3 (TF32 off), 1.49e-3 (TF32 on), 100 images. Its device
  memory (960 MiB) is the pcb1 row; candle was not timed.
- **Steel detector.** On the selection split the augmented Faster R-CNN has the lower cost at all three ratios, with the
  PyTorch path (1004 / 1596 / 4344 against D-FINE-S 1169 / 1888 / 4958) and with ORT TF32 off (1033 / 1619 / 4448 against
  1185 / 1914 / 4959). Both costs are optimistic by construction (`val` also picked each checkpoint). D-FINE-S leads on val
  AP@50:95 (0.4346 against 0.3996, `reports/detection_comparison.md`), has the smaller footprint (560-572 against 1,278-1,332 MiB) and
  runs ORT at 30.9 against 77.0 ms. I had read the `test` numbers of Phase 2 and 4a before choosing, and they point the
  same way for cost (Faster R-CNN lowest at every ratio), so they are not an independent confirmation. The two ONNX contracts
  are both implemented in the API (`format: torchvision` or `dfine`); switching is the `steel` block of the config:
  `format: dfine`, `onnx: models/onnx/dfine_s.onnx`, `frozen_model: dfine_s`, `vram_mib: 572`, `tf32: false`.
- **Not fused across domains:** a `visa_*` product never runs the steel detector and `steel` never runs an anomaly model.
  The detector was trained on 200 x 200 steel images, so `steel` warns when the upload has another size.
- **Known weak product:** `visa_macaroni2` (EfficientAD, calibration image AUROC 0.8268) has a frozen threshold that flags 44 of
  the 50 normal calibration images (`reports/thresholds.md`, sel FN/FP 0/44 at every ratio), and in the live check a normal
  macaroni2 image was flagged. A threshold cannot fix a weak ranking (same note as Phase 4a). It is served so that every
  VisA category has a product; its flags carry little information.

## 2. Do the frozen thresholds survive the move to ONNX Runtime?

Reference: the Phase 4a PyTorch path, after it reproduced the saved scores (VisA, max diff 0) and the stored val metrics
(detectors, max diff 0). Other side: the serving code on ORT CUDA (byte decoding, preprocessing, session options and detector
decoding as in the API), TF32 off and on. Selection splits only: `threshold_calibration` (75-100 images per category) and
`val` (270 images).

- **VisA: no image changes side.** 0 flips at every threshold, for all four products, TF32 off and on (375 images, 3 ratios,
  2 settings; the three EfficientAD categories have one threshold for all ratios).
- **This is weaker evidence than it looks.** The nearest image to a threshold is 0.27-1.61% (relative, over all thresholds) away under ORT, while the
  largest score difference between the paths is 0.15-1.41% (EfficientAD cashew, TF32 off: 1.41e-2, nearest image to the threshold 0.30%;
  PatchCore candle, ratio 10 threshold: nearest image 0.27%). The
  PyTorch reference is itself the cuDNN-default path (TF32 convolutions), about 1% from strict fp32 (Phase 5a). So zero flips
  is a fact about these 375 images, not a bound: an unseen image within about 1% of a threshold can land on the other side of
  it in ORT. Those are the images the threshold is least sure about anyway, but the count of them in production is unmeasured.
- **Detectors: no image changes alarm status, but boxes near the threshold move.** Every val image keeps at least one box at
  every threshold (NEU-DET has no defect-free images), so the alarm flips are 0 by construction and mean nothing. What changes
  is the number of boxes kept: Faster R-CNN, TF32 off, 13 / 17 / 66 of 270 images at ratio 5 / 10 / 50, and the val cost at the
  frozen threshold goes from 1004 / 1596 / 4344 to 1033 / 1619 / 4448 (+2.9% / +1.4% / +2.4%). TF32 on: 9 / 14 / 43 images,
  +1.5% / +1.2% / +1.4%. D-FINE-S (not served): 22 / 31 / 82 images, +1.4% / +1.4% / +0.0% (TF32 off).
- **Part of that is the reference, not ORT.** The detector thresholds were chosen on fp16-autocast scores (the Phase 2 path),
  and Phase 5a found AMP as far from strict fp32 as ORT with TF32 on in raw outputs. The ORT fp32 scores are arguably the
  more accurate ones; the frozen thresholds were just tuned on the noisier path. This run can not separate the two.

**How I handle it (decided, not just proposed):** keep the frozen thresholds unchanged and serve them, because they are
what Phase 4a evaluated and the evaluation split was read once. The API returns the threshold, the cost ratio, the provider,
the TF32 setting and the frozen file's SHA-256 with every answer, so a result can be traced to the operating point. The
steel cost shift above is disclosed here and in the README, not corrected. **Alternative I did not take:** re-select the
thresholds on ORT scores of the selection splits. That gives thresholds matching what the API computes, but they would be new
numbers no evaluation has seen, and checking them on `threshold_evaluation` / `test` would be a second read of those splits
(a protocol breach under the Phase 4a rules). **Decided by the project owner after reading this: not taken, see "Decisions" below.**

**Not measured:** the CPU fallback path (ORT CPU against PyTorch); any evaluation split; false alarms on clean steel (there is
no clean steel); the number of production images near a threshold. The API answers on CPU when CUDA is unavailable, and the
thresholds have not been checked for that path.

## 3. TF32 per served model

The rule (in `export/serve_report.py`, fixed before counting flips): TF32 on only if there are no flips with it on and it saves at
least 20% of the model's p50 latency in the Phase 5a benchmark. The flip definition for detectors (images with a different number
of kept boxes) and the 20% are mine. Result: **steel off** (9 / 14 / 43 images with a different box count at the three ratios with TF32 on, 66 image-ratio pairs,
although it would be 40% faster, 45.94 against 77.01 ms), **EfficientAD on** (0 flips, 30%: 9.03 against 12.95 ms, benchmark row
of pcb1 used for all three categories), **PatchCore off** (0 flips but 12%: 9.20 against 10.44 ms). D-FINE-S, if switched in, stays
off as decided in Phase 5a. A test checks that the config follows the rule on the committed results.

- The saving is model latency only. In the live check an EfficientAD request took 27.6 ms in total of which 9.3 ms was the model
  (the rest is decoding, preprocessing and the PNG heatmap), so TF32 is worth about 4 ms of 28 ms there.
- Steel with TF32 on was slightly closer to the AMP-calibrated thresholds (cost shift +1.5% / +1.2% / +1.4% against +2.9% / +1.4% /
  +2.4%), one run each and no interval, so I do not read it as a difference. `tf32: true` in the `steel` block turns it on.

## Live check (WSL2, RTX 4060 Laptop, one client, sequential `curl`)

- **A bug the unit tests could not see:** in the first live run the `steel` session ran on **CPU** while `/health` said CUDA.
  Without torch imported first, ORT could not find `libcublasLt.so.13` (the Phase 5a scripts import torch, which loads it), and
  it drops the provider silently; `get_available_providers()` still lists CUDA. Fixed: `onnxruntime.preload_dlls()` and a probe
  session at start-up, so `/health` reports the provider that works and the reason when it does not
  (`test_cuda_in_the_build_but_not_loadable_is_reported_not_assumed`). The consistency counts above were run in a process with
  torch imported and report `CUDAExecutionProvider` for every row.
- All five products answered 200 on a selection-split image, `provider` `CUDAExecutionProvider` in every response. With all five
  loaded, device-wide `memory.used` read 3,492 MiB (970 MiB read earlier with the GPU idle and no server, one reading each).
- Warm latency, 30 calls after 3 warm-up calls, server-side total p50 / p95: steel 75.2 / 77.3 ms (model 73.5 ms), pcb1 27.6 / 28.2
  ms, candle 31.0 / 32.3 ms. Client wall time with a new `curl` per call: 85.1 / 87.2, 39.2 / 40.0, 41.6 / 43.3 ms. First call
  per product: model load 62-573 ms and a slower first inference (steel 682 ms).
- **Limits:** a single client; no concurrency or load test; no authentication or rate limiting; a model load holds the registry lock,
  so other requests wait for it (hundreds of ms); heatmap regions are derived from the map and not evaluated as boxes (only pixel
  AUROC and AUPRO were measured); the `fn_fp_ratio` choices are the assumed ratios of Phase 4a, not measured costs.

## Decisions (project owner, after reading the Phase 6a report; 2026-11-08)

1. **The frozen Phase 4a thresholds stay as they are. No re-selection on ORT scores.** The API serves the thresholds in
   `reports/frozen_thresholds.json` for every product and returns the operating point with each answer. The steel cost shift
   (+2.9% / +1.4% / +2.4% on `val`, section 2) and the zero-flip VisA result with its ~1% caveat remain disclosed limitations,
   not something to correct. Consequence: the evaluation splits are not read a second time.
2. **`steel` stays the augmented Faster R-CNN** (`format: torchvision`, `models/onnx/faster_rcnn.onnx`). D-FINE-S remains
   implemented behind the config switch in section 1 but is not served. The decision did not mention the 0.05 AUROC rule for the anomaly products, so it stands
   as written (my judgement, not a confirmed choice) (cashew EfficientAD, candle PatchCore).

This section records the decisions; no measurement changed.
